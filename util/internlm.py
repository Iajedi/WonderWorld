import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import openai
import requests
import spacy
from openai import OpenAI
from PIL import Image as PILImage

# run 'python -m spacy download en_core_web_sm' to load english language model
nlp = spacy.load("en_core_web_sm")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "NOT_USED_FOR_INTERNLM"),
    base_url='http://0.0.0.0:23333/v1'
)


def _parse_json_object_text(raw):
    raw = raw.strip()
    return json.loads(raw)


def _normalize_scene_dict(output):
    if isinstance(output, tuple):
        output = output[0]
    _ = output["scene_name"], output["entities"], output["background"]
    if isinstance(output["scene_name"], str):
        output["scene_name"] = [output["scene_name"]]
    if isinstance(output["entities"], str):
        output["entities"] = [output["entities"]]
    if isinstance(output["background"], str):
        output["background"] = [output["background"]]
    return output


def _yes_no_from_text(text):
    t = text.strip().lower()
    for prefix in ("yes", "no"):
        if t.startswith(prefix):
            return prefix
    words = t.split()
    if words and words[0] in ("yes", "no"):
        return words[0]
    if "yes" in t and "no" not in t[:20]:
        return "yes"
    if "no" in t and "yes" not in t[:20]:
        return "no"
    return None


class TextpromptGen(object):

    def __init__(self, root_path, control=False):
        super(TextpromptGen, self).__init__()
        self.model = client.models.list().data[0].id
        self.vision_model = client.models.list().data[0].id
        self.save_prompt = True
        self.scene_num = 0
        self.base_url = client.base_url
        self.id = 0
        if control:
            self.base_content = "Please generate scene description based on the given information:"
        else:
            self.base_content = "Please generate next scene based on the given scene/scenes information:"
        self.content = self.base_content
        self.root_path = root_path
        self.destination_output = None
        self.init_scene_dict = None
        self._last_resolved = None

    def get_destination_output(self):
        return self.destination_output

    def set_destination_output(self, scene_dict):
        self.destination_output = {
            "scene_name": scene_dict["scene_name"],
            "entities": scene_dict["entities"],
            "background": scene_dict["background"],
        }

    def write_json(self, output, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            output["background"][0] = self.generate_keywords(output["background"][0])
            with open(save_dir / "scene_{}.json".format(str(self.scene_num).zfill(2)), "w") as json_file:
                json.dump(output, json_file, indent=4)
        except Exception:
            pass
        return

    def write_all_content(self, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "all_content.txt", "w") as f:
            f.write(self.content)
        return

    def _chat_completion_text(self, messages, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def regenerate_background(self, style, entities, scene_name, background=None):
        if background is not None:
            content = (
                "Please generate a brief scene background with Scene name: "
                + scene_name
                + "; Background: "
                + str(background).strip(".")
                + ". Entities: "
                + str(entities)
                + "; Style: "
                + str(style)
            )
        else:
            content = (
                "Please generate a brief scene background with Scene name: "
                + scene_name
                + "; Entities: "
                + str(entities)
                + "; Style: "
                + str(style)
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent scene generator. Given a scene and there are 3 most significant common "
                    "entities. please generate a brief background prompt about 20 words describing common things in "
                    "the scene. You should not mention the blank space, artifacts or entities in the background prompt. "
                    "If needed, you can make reasonable guesses. Reply with only the background text, no JSON and no "
                    "extra commentary."
                ),
            },
            {"role": "user", "content": content},
        ]
        background = self._chat_completion_text(messages)
        return background.strip().strip(".")

    def set_initial_resolved_state(self, scene_dict, style=None):
        if scene_dict is None:
            return
        sn = scene_dict["scene_name"]
        self._last_resolved = {
            "scene_name": [sn] if isinstance(sn, str) else list(sn),
            "entities": [scene_dict["entities"]] if isinstance(scene_dict["entities"], str) else list(scene_dict["entities"]),
            "background": (
                [scene_dict["background"]]
                if scene_dict.get("background") is not None and not isinstance(scene_dict["background"], list)
                else (list(scene_dict["background"]) if scene_dict.get("background") is not None else [""])
            ),
        }
        if style is not None:
            self._style_for_initial = style

    def _resolved_dict_for_snapshot(self, output):
        return {
            "scene_name": list(output["scene_name"]) if not isinstance(output["scene_name"], str) else [output["scene_name"]],
            "entities": list(output["entities"]) if not isinstance(output["entities"], str) else [output["entities"]],
            "background": list(output["background"]) if not isinstance(output["background"], str) else [output["background"]],
        }

    def _scene_dict_for_bcdm(self, scene_dict):
        return _normalize_scene_dict(dict(scene_dict))

    def _scene_user_spec_block(self, style, d, worldscore=False):
        d = self._scene_dict_for_bcdm(d)
        name = d["scene_name"]
        name0 = name[0] if isinstance(name, (list, tuple)) else str(name)
        ent = d["entities"]
        bg = d["background"]
        # Patch for worldscore
        if worldscore:
            raw_name = name[0].split(" ")[0]
            return f"Image description: {raw_name}, style: {style}"
        bg0 = bg[0] if isinstance(bg, (list, tuple)) and bg else bg
        return (
            f"Image style: {style}\n"
            f"Target scene name: {name0}\n"
            f"Entities: {ent!s}\n"
            f"Background: {bg0}\n"
        )

    def wonder_next_scene(self, style=None, entities=None, scene_name=None, background=None, change_scene_name_by_user=False):
        if change_scene_name_by_user:
            self.scene_num += 1
            self.id += 1
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
        elif style is not None and entities is not None:
            assert not (background is None and scene_name is None), "At least one of the background and scene_name should not be None"
            self.scene_num += 1
            self.id += 1
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            scene_content = (
                "\nScene " + str(self.id) + ": {Scene name: " + str(scene_name).strip(".") + "; Entities: " + str(entities) + "; Style: " + str(style) + "}"
            )
            self.content += scene_content
        else:
            assert self.scene_num > 0, "To regenerate the scene description, you should have at least one scene content as prompt."

        if change_scene_name_by_user:
            system = (
                "You are an intelligent scene generator. Imagine you are wandering through a scene or a sequence of scenes, "
                "and there are 3 most significant common entities in each scene. The next scene you would go to is "
                + str(scene_name)
                + ". Please generate the corresponding 3 most common entities in this scene. The scenes are sequentially "
                "interconnected, and the entities within the scenes are adapted to match and fit with the scenes. You also "
                "have to generate a brief background prompt about 50 words describing the scene. You should not mention the "
                "entities in the background prompt. If needed, you can make reasonable guesses. "
                "You MUST respond with a single JSON object only, no markdown, in exactly this format:\n"
                "{'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"
            )
        else:
            system = (
                "You are an intelligent scene generator. Imagine you are wandering through a scene or a sequence of scenes, "
                "and there are 3 most significant common entities in each scene. Please tell me what sequential next scene "
                "you would most likely see. You need to generate the scene name and the 3 most common entities in the scene. "
                "The scenes are sequentially interconnected, and the entities within the scenes are adapted to match and fit "
                "with the scenes. You also have to generate a brief background prompt about 50 words describing the scene. "
                "You should not mention the entities in the background prompt. If needed, you can make reasonable guesses. "
                "You MUST respond with a single JSON object only, no markdown, in exactly this format:\n"
                "{'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2', 'entity_3'], 'background': ['background prompt']}"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.content},
        ]

        output = None
        for _ in range(10):
            try:
                response = self._chat_completion_text(messages, response_format={"type": "json_object"})
                print(response)
                try:
                    output = _parse_json_object_text(response)
                    output = _normalize_scene_dict(output)
                    break
                except Exception as e:
                    assistant_message = {"role": "assistant", "content": response}
                    user_message = {
                        "role": "user",
                        "content": (
                            "The output is not valid JSON. Reply with a single JSON object only, no markdown, matching: "
                            "{'scene_name': ['...'], 'entities': ['e1','e2','e3'], 'background': ['...']}\n"
                            f"Previous invalid reply: {response}\n"
                            f"Error: {e!s}"
                        ),
                    }
                    messages.append(assistant_message)
                    messages.append(user_message)
                    print("OpenAI JSON parse failed, retrying.", str(e))
                    continue
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print("Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue

        if output is not None:
            self._last_resolved = self._resolved_dict_for_snapshot(output)

        if self.save_prompt and output is not None:
            self.write_json(output)
            self.write_all_content()

        return output

    def generate_keywords(self, text):
        doc = nlp(text)

        adj = False
        noun = False
        text_out = ""
        for token in doc:
            if token.pos_ not in ("NOUN", "ADJ"):
                continue

            if token.pos_ == "NOUN":
                if adj:
                    text_out += " " + token.text
                    adj = False
                    noun = True
                else:
                    if noun:
                        text_out += ", " + token.text
                    else:
                        text_out += token.text
                        noun = True
            elif token.pos_ == "ADJ":
                if adj:
                    text_out += " " + token.text
                else:
                    if noun:
                        text_out += ", " + token.text
                        noun = False
                        adj = True
                    else:
                        text_out += token.text
                        adj = True

        return text_out

    def generate_prompt(self, style, entities, background=None, scene_name=None):
        assert not (background is None and scene_name is None), "At least one of the background and scene_name should not be None"
        if background is not None:
            if isinstance(background, list):
                background = background[0]

            background = self.generate_keywords(background)
            prompt_text = "Style: " + style + ". Entities: "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                else:
                    prompt_text += ", " + entity
            prompt_text += ". Background: " + background
            print("PROMPT TEXT: ", prompt_text)
        else:
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            prompt_text = "Style: " + style + ". " + scene_name + " with "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                elif i == len(entities) - 1:
                    prompt_text += ", and " + entity
                else:
                    prompt_text += ", " + entity
            print("PROMPT TEXT: ", prompt_text)

        return prompt_text

    def encode_image_pil(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _conditioning_image_to_pil(self, image):
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        try:
            import torch

            if isinstance(image, torch.Tensor):
                x = image.detach().float().cpu()
                if x.dim() == 4:
                    x = x[0]
                x = x.clamp(0, 1).numpy()
                arr = (np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8)
                return PILImage.fromarray(arr)
        except ImportError:
            pass
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 1)
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            if image.ndim == 2:
                raise ValueError("Expected HWC RGB array for conditioning image")
            return PILImage.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported conditioning image type: {type(image)}")

    def _vision_completion_text(self, prompt, image, system=None, max_tokens=300):
        base64_image = self.encode_image_pil(self._conditioning_image_to_pil(image))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        )
        payload = {
            "model": self.vision_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            str(self.base_url) + "chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def build_bcdm_inpaint_pair_from_conditioning_image(self, conditioning_image, style, scene_dict, worldscore=False):
        pil = self._conditioning_image_to_pil(conditioning_image)
        d = self._scene_dict_for_bcdm(scene_dict)
        user_spec = self._scene_user_spec_block(style, d, worldscore=worldscore)

        system_src = (
            "You describe images for inpainting conditioning. Clearly list what is visible in the image."
            "Do not mention blank space, artifacts or invent occluded content. "
            "Reply with plain prose only, 1 - 2 short sentences, no JSON or bullet labels.\n"
        )
        prompt_src = ""
        for _ in range(3):
            try:
                prompt_src = self._vision_completion_text(
                    "Describe this frame for an inpainting editor. "
                    "STRICTLY at most 2 sentences, concise and precise, preserving only essential visible details. "
                    "Example: A DSLR 35mm landscape view, set in the late afternoon sun with long shadows, shows the existing paved town square with its fountain, reflections, line of trees, and people at outdoor cafe tables next to a building on the left, with the prominent clock tower with a dark dome roof and other orange-roofed buildings in the background.",
                    pil,
                    system=system_src,
                )
                if prompt_src:
                    print("PROMPT TEXT (BCDM prompt_src, vision): ", prompt_src)
                    break
            except Exception as e:
                print(f"OpenAI BCDM prompt_src (vision) retry: {e}")
                time.sleep(0.5)
        if not prompt_src:
            prompt_src = (
                "Foreground, background, and style as in the conditioning render: preserve visible geometry, objects, "
                "and sky regions that are already drawn."
            )

        # --- Call 2 (text): prompt_tgt ---
        # Prefill: assistant message already contains (A); model continues in-place (suffix only).
        system_tgt = (
            "You write one English text-to-image description to extend a scene. "
            "You are given (A) an accurate description of what is already visible in the frame and (B) a target "
            "scene specification (style, scene name, entities, background). "
            "Produce 1-2 fluent sentences that: summarizes (A) and elaborates (B) "
            "by describing the spatial placement of EACH OBJECT in (B) relative to (A). "
            "Output only the final prompt, no title or preamble. NO COMMAND WORDS.\n"
        )
        user_tgt = (
            "Source frame description (A):\n"
            f"{prompt_src}\n\n"
            "Target scene specification (B):\n"
            f"{user_spec}\n\n"
            "Write the description of the scene, describing each object in (B) relative to (A)."
            "STRICTLY at most 2 sentences, DO NOT INVENT ANYTHING, preserving only essential visible details."
            "Example: Behind the orange-roofed building is a city street with sidewalks, featuring a bus stop, waiting passengers, and a traffic light, as a bus travels along the steady flow of vehicles and pedestrians, embodying vibrant urban activity."
            )
        prompt_tgt = ""
        for _ in range(3):
            try:
                prompt_tgt = self._chat_completion_text(
                    [
                        {"role": "system", "content": system_tgt},
                        {"role": "user", "content": user_tgt}
                    ]
                ).strip()
                if prompt_tgt:
                    print("PROMPT TEXT (BCDM prompt_tgt, text): ", prompt_tgt)
                    break
            except Exception as e:
                print(f"OpenAI BCDM prompt_tgt retry: {e}")
                time.sleep(0.5)
        if not prompt_tgt:
            sn = d["scene_name"]
            sn0 = sn[0] if isinstance(sn, (list, tuple)) else str(sn)
            ent = d["entities"]
            if isinstance(ent, str):
                ent = [ent]
            bg = d["background"]
            prompt_tgt = self.generate_prompt(style, list(ent), background=bg, scene_name=sn0)

        return prompt_src, prompt_src + " " + prompt_tgt

    def _describe_image_for_edit(self, image, system, prompt, fallback):
        pil = self._conditioning_image_to_pil(image).resize((512, 512))
        for _ in range(3):
            try:
                text = self._vision_completion_text(prompt, pil, system=system)
                if text:
                    print("PROMPT TEXT (edit image description): ", text)
                    return text
            except Exception as e:
                print(f"OpenAI edit image description retry: {e}")
                time.sleep(0.5)
        return fallback

    def describe_edit_source_without_mask(self, masked_source_image):
        system = (
            "You describe images for object-removal inpainting. The image may contain a black blanked area where "
            "an object or region was removed. Describe only the visible content OUTSIDE that blanked area "
            "(scene layout, materials, lighting, background, nearby context) so the follow-up inpaint can produce a "
            "seamless transition into the removed region. Do not describe or guess the removed object. "
            "STRICTLY at most 2 sentences, as concise and precise as possible, preserving only essential visible "
            "details. No title, no bullets, no preamble."
        )
        prompt = (
            "Describe the image background ONLY. "
            "Exclude the missing region and do not invent what was there. "
            "STRICTLY at most 2 concise, precise sentences with essential details only."
        )
        fallback = (
            "A detailed scene with the visible regions preserved around a blanked edit area; describe the background, "
            "surfaces, lighting, and surrounding objects while excluding the removed region."
        )
        return self._describe_image_for_edit(masked_source_image, system, prompt, fallback)

    def describe_composed_edit_image(self, composed_image):
        system = (
            "You describe edited images for a follow-up refinement model. Describe the full composed scene exactly "
            "as visible (inserted/transformed foreground, surrounding context, lighting, materials) so refinement can "
            "produce a seamless transition between edited and untouched regions. "
            "STRICTLY at most 2 sentences, as concise and precise as possible, preserving only essential details. "
            "No title, no bullets, no preamble."
        )
        prompt = (
            "Describe this composed/edited image for refinement so it blends seamlessly with surrounding regions. "
            "STRICTLY at most 2 concise, precise sentences with essential details only."
        )
        fallback = (
            "A detailed composed scene containing the edited foreground and surrounding background, with coherent "
            "lighting, placement, materials, and scene context."
        )
        return self._describe_image_for_edit(composed_image, system, prompt, fallback)

    def _multimodal_yes_no(self, image, question):
        prompt = question + " Your answer should be simply 'Yes' or 'No'."
        for _ in range(5):
            try:
                text = self._vision_completion_text(prompt, image)
                yn = _yes_no_from_text(text) if text else None
                if yn in ("yes", "no"):
                    return yn
            except Exception as e:
                print(f"OpenAI vision error, retrying: {e}")
                time.sleep(1)
        return None

    def evaluate_image(self, image, eval_blur=True, eval_partial=False):
        border_text = (
            "Along the four borders of this image, is there anything that looks like thin border, thin stripe, "
            "photograph border, painting border, or painting frame? Please look very closely to the four edges and try hard, "
            "because the borders are very slim and you may easily overlook them. I would lose my job if there is a border and "
            "you overlook it. If you are not sure, then please say yes."
        )
        print(border_text)
        has_border = True
        border = self._multimodal_yes_no(image, border_text)
        if border in ("yes", "no"):
            print("Border: ", border)
            has_border = border == "yes"

        if eval_blur:
            blur_text = (
                "Does this image have a significant blur issue or blurry effect caused by out of focus around the image "
                "edges? You only have to pay attention to the four borders of the image."
            )
            print(blur_text)
            has_blur = True
            blur = self._multimodal_yes_no(image, blur_text)
            if blur in ("yes", "no"):
                print("Blur: ", blur)
                has_blur = blur == "yes"
        else:
            has_blur = False

        if eval_partial:
            partial_text = (
                "Does this image have any objects that are only partially visible against the sky? Please look very "
                "closely. If you are not sure, then please say yes."
            )
            print(partial_text)
            has_partial = True
            part = self._multimodal_yes_no(image, partial_text)
            if part in ("yes", "no"):
                print("Partial: ", part)
                has_partial = part == "yes"
        else:
            has_partial = False
            
        return has_border, has_blur, has_partial
