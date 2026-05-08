"""
Gemini-backed scene prompt generator: drop-in alternative to :class:`util.chatGPT4.TextpromptGen`
for next-scene JSON, background regeneration, and image QA (replaces GPT-4 / GPT-4V flows).

**Does not** import or modify ``chatGPT4.py``. Uses the same public method shapes so callers
can switch with one import change, for example:

    from util.gemini_prompt_gen import GeminiTextpromptGen as TextpromptGen

**Setup**

* ``pip install google-generativeai`` (add to your environment; not required if you do not use this module).
* Set ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` to a Google AI Studio / Gemini API key.
* For :meth:`generate_keywords` / :meth:`generate_prompt`, the same spacy model as
  ``chatGPT4`` is used: ``python -m spacy download en_core_web_sm``.
* BCOT (FLUX) prompts: call :meth:`build_bcot_inpaint_pair_from_conditioning_image` once per inpaint
  with the conditioning render tensor/PIL — **two** Gemini calls (vision ``prompt_src``, text ``prompt_tgt``).
  Scene JSON still uses :meth:`wonder_next_scene` separately when ``use_gpt`` is enabled.
"""
from __future__ import annotations

import base64
import io
import json
import time
import os
import numpy as np
from pathlib import Path

import spacy

# Same loading pattern as util/chatGPT4.py
nlp = spacy.load("en_core_web_sm")

try:
    import google.generativeai as genai
except ImportError as e:  # pragma: no cover
    genai = None
    _GENAI_IMPORT_ERROR = e
else:
    _GENAI_IMPORT_ERROR = None


def _require_genai():
    if genai is None:
        raise ImportError(
            "google-generativeai is required for GeminiTextpromptGen. "
            f"Import failed: {_GENAI_IMPORT_ERROR}. "
            "Install with: pip install google-generativeai"
        ) from _GENAI_IMPORT_ERROR
    return genai


def _parse_json_object_text(raw: str) -> dict:
    """Parse model output as a JSON object (avoids eval)."""
    raw = raw.strip()
    return json.loads(raw)


def _normalize_scene_dict(output: dict) -> dict:
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


def _yes_no_from_text(text: str) -> str | None:
    """Extract 'yes' or 'no' from a short model reply."""
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


class GeminiTextpromptGen:
    """
    Parity with :class:`util.chatGPT4.TextpromptGen` for methods used in scene and image checks.

    LLM calls use the Gemini API; keyword/prompt assembly matches ``TextpromptGen``.
    """

    def __init__(
        self,
        root_path,
        control: bool = False,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ):
        _require_genai()
        self.model = model
        self.save_prompt = True
        self.scene_num = 0
        self.id = 0
        if control:
            self.base_content = "Please generate scene description based on the given information:"
        else:
            self.base_content = "Please generate next scene based on the given scene/scenes information:"
        self.content = self.base_content
        self.root_path = root_path
        self.destination_output = None
        self.init_scene_dict = None

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GeminiTextpromptGen requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment "
                "(or pass api_key=)."
            )
        genai.configure(api_key=key)
        # Previous resolved scene (after each successful wonder_next_scene)
        self._last_resolved: dict | None = None

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
            with open(save_dir / f"scene_{str(self.scene_num).zfill(2)}.json", "w") as json_file:
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

        system = (
            "You are an intelligent scene generator. Given a scene and there are 3 most significant common entities. "
            "please generate a brief background prompt about 50 words describing common things in the scene. "
            "You should not mention the entities in the background prompt. If needed, you can make reasonable guesses. "
            "Reply with only the background text, no JSON and no extra commentary."
        )
        g = _require_genai()
        gmodel = g.GenerativeModel(
            self.model,
            system_instruction=system,
        )
        r = gmodel.generate_content(content)
        return (r.text or "").strip().strip(".")

    def set_initial_resolved_state(self, scene_dict, style: str | None = None) -> None:
        """Call once with the initial ``scene_dict`` from the example YAML (tracks prior scene for continuity)."""
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

    def _resolved_dict_for_snapshot(self, output: dict) -> dict:
        return {
            "scene_name": list(output["scene_name"])
            if not isinstance(output["scene_name"], str)
            else [output["scene_name"]],
            "entities": list(output["entities"])
            if not isinstance(output["entities"], str)
            else [output["entities"]],
            "background": list(output["background"])
            if not isinstance(output["background"], str)
            else [output["background"]],
        }

    def _scene_dict_for_bcot(self, scene_dict: dict) -> dict:
        """Normalize scene_dict for BCOT target elaboration (same keys as wonder_next_scene JSON)."""
        return _normalize_scene_dict(dict(scene_dict))

    def _scene_user_spec_block(self, style: str, d: dict) -> str:
        """Human-readable block for the target-scene / user-intent side of BCOT."""
        d = self._scene_dict_for_bcot(d)
        name = d["scene_name"]
        name0 = name[0] if isinstance(name, (list, tuple)) else str(name)
        ent = d["entities"]
        bg = d["background"]
        bg0 = bg[0] if isinstance(bg, (list, tuple)) and bg else bg
        return (
            f"Image style: {style}\n"
            f"Target scene name: {name0}\n"
            f"Entities: {ent!s}\n"
            f"Background: {bg0}\n"
        )

    def _conditioning_image_to_pil(self, image):
        """Accept PIL Image or CHW float tensor [0,1] (optional batch dim)."""
        from PIL import Image as PILImage

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
                raise ValueError("Expected HWC RGB array for BCOT conditioning image")
            return PILImage.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported conditioning image type: {type(image)}")

    def build_bcot_inpaint_pair_from_conditioning_image(
        self,
        conditioning_image,
        style: str,
        scene_dict: dict,
    ) -> tuple[str, str]:
        """BCOT prompts for one inpaint step: exactly **two** Gemini calls.

        1. **Vision** — ``prompt_src``: describe foreground and background as visible in the conditioning render.
        2. **Text** — ``prompt_tgt``: one fluent prompt merging that observation with an elaboration of the target
           scene (``scene_dict``: scene name, entities, background) and ``style``.

        On failure, falls back to a heuristic ``prompt_src`` and :meth:`generate_prompt`-style ``prompt_tgt``.
        """
        _require_genai()
        pil = self._conditioning_image_to_pil(conditioning_image)
        d = self._scene_dict_for_bcot(scene_dict)
        user_spec = self._scene_user_spec_block(style, d)

        # --- Call 1 (vision): prompt_src ---
        system_src = (
            "You describe images for inpainting conditioning. Clearly separate **foreground** (main subjects, "
            "nearby objects, terrain, structures in front) and **background** (sky, horizon, distant scenery). "
            "Only state what is visibly present; do not invent occluded content. "
            "Reply with plain prose only, 2 - 5 short sentences, no JSON or bullet labels."
        )
        g = _require_genai()
        model_src = g.GenerativeModel(self.model, system_instruction=system_src)
        prompt_src = ""
        for _ in range(3):
            try:
                r = model_src.generate_content(
                    [
                        "Describe the foreground and the background of this image as they would appear to someone "
                        "editing or inpainting this frame.",
                        pil,
                    ]
                )
                prompt_src = (r.text or "").strip()
                if prompt_src:
                    print("PROMPT TEXT (BCOT prompt_src, vision): ", prompt_src)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"Gemini BCOT prompt_src (vision) retry: {e}")
                time.sleep(0.5)
        if not prompt_src:
            prompt_src = (
                "Foreground and background as in the conditioning render: preserve visible geometry, objects, "
                "and sky regions that are already drawn."
            )

        # --- Call 2 (text): prompt_tgt ---
        system_tgt = (
            "You write one English text-to-image prompt for boundary-conditioned inpainting. "
            "You are given (A) an accurate description of what is already visible in the frame and (B) a target "
            "scene specification (style, scene name, entities, background). "
            "Produce a single fluent paragraph that: keeps (A) as the anchor for observed regions; elaborates (B) "
            "so newly generated / inpainted areas match the intended scene; does not contradict (A). "
            "Output only the final prompt, no title or preamble."
        )
        user_tgt = (
            f"{user_spec}\n"
            f"--- What is already visible in the frame (do not contradict this) ---\n{prompt_src}\n"
        )
        model_tgt = g.GenerativeModel(self.model, system_instruction=system_tgt)
        prompt_tgt = ""
        for _ in range(3):
            try:
                r = model_tgt.generate_content(user_tgt)
                prompt_tgt = (r.text or "").strip()
                if prompt_tgt:
                    print("PROMPT TEXT (BCOT prompt_tgt, text): ", prompt_tgt)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"Gemini BCOT prompt_tgt retry: {e}")
                time.sleep(0.5)
        if not prompt_tgt:
            sn = d["scene_name"]
            sn0 = sn[0] if isinstance(sn, (list, tuple)) else str(sn)
            ent = d["entities"]
            if isinstance(ent, str):
                ent = [ent]
            bg = d["background"]
            prompt_tgt = self.generate_prompt(style, list(ent), background=bg, scene_name=sn0)

        return prompt_src, prompt_tgt

    def describe_edit_source_without_mask(self, masked_source_image) -> str:
        """Describe the visible scene after the selected edit region has been blanked."""

        system = (
            "You describe images for object-removal inpainting. The image may contain a black blanked area where "
            "an object or region was removed. Describe only the visible content outside that blanked area, including "
            "scene layout, materials, lighting, background, and nearby context. Do not describe or guess the removed "
            "object. Output one detailed paragraph, no title or bullets."
        )
        prompt = (
            "Describe the image content outside the blanked/removed region. Exclude the missing region and do not "
            "invent what was there."
        )
        fallback = (
            "A detailed scene with the visible regions preserved around a blanked edit area; describe the background, "
            "surfaces, lighting, and surrounding objects while excluding the removed region."
        )
        return self._describe_image_for_edit(masked_source_image, system, prompt, fallback)

    def describe_composed_edit_image(self, composed_image) -> str:
        """Describe the composed image after the first geometry/FLUX pass."""

        system = (
            "You describe edited images for a follow-up refinement model. Describe the full composed scene exactly "
            "as visible, including the inserted or transformed foreground, surrounding context, lighting, materials, "
            "and how the edit fits into the scene. Output one detailed paragraph, no title or bullets."
        )
        prompt = "Describe this composed/edited image in detail for image refinement."
        fallback = (
            "A detailed composed scene containing the edited foreground and surrounding background, with coherent "
            "lighting, placement, materials, and scene context."
        )
        return self._describe_image_for_edit(composed_image, system, prompt, fallback)

    def _describe_image_for_edit(self, image, system: str, prompt: str, fallback: str) -> str:
        _require_genai()
        pil = self._conditioning_image_to_pil(image).resize((512, 512))
        g = _require_genai()
        model = g.GenerativeModel(self.model, system_instruction=system)
        for _ in range(3):
            try:
                r = model.generate_content([prompt, pil])
                text = (r.text or "").strip()
                if text:
                    print("PROMPT TEXT (edit image description): ", text)
                    return text
            except Exception as e:  # noqa: BLE001
                print(f"Gemini edit image description retry: {e}")
                time.sleep(0.5)
        return fallback

    def wonder_next_scene(
        self, style=None, entities=None, scene_name=None, background=None, change_scene_name_by_user=False
    ):
        if change_scene_name_by_user:
            self.scene_num += 1
            self.id += 1
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
        elif style is not None and entities is not None:
            assert not (background is None and scene_name is None), (
                "At least one of the background and scene_name should not be None"
            )
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

        g = _require_genai()
        generation_config = g.GenerationConfig(
            response_mime_type="application/json",
        )
        current_user = self.content
        output = None

        for _ in range(10):
            try:
                gmodel = g.GenerativeModel(
                    self.model,
                    system_instruction=system,
                )
                r = gmodel.generate_content(
                    current_user,
                    generation_config=generation_config,
                )
                response = r.text or ""
                print(response)
                try:
                    output = _parse_json_object_text(response)
                    output = _normalize_scene_dict(output)
                    break
                except Exception as e:  # noqa: BLE001
                    repair = (
                        "The output is not valid JSON. Reply with a single JSON object only, no markdown, matching: "
                        "{'scene_name': ['...'], 'entities': ['e1','e2','e3'], 'background': ['...']}\n"
                        f"Previous invalid reply: {response}\n"
                        f"Error: {e!s}"
                    )
                    current_user = self.content + "\n\n" + repair
                    print("Gemini JSON parse failed, retrying.", str(e))
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"Gemini API error: {e}")
                print("Wait for a second and retry wonder_next_scene.")
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

    def _multimodal_yes_no(self, image, question: str) -> str | None:
        """Ask a yes/no question about a PIL image; return 'yes', 'no', or None if unclear."""
        g = _require_genai()
        model = g.GenerativeModel(self.model)
        prompt = question + " Your answer should be simply 'Yes' or 'No'."
        for _ in range(5):
            try:
                r = model.generate_content([prompt, image])
                text = r.text
                yn = _yes_no_from_text(text) if text else None
                if yn in ("yes", "no"):
                    return yn
            except Exception as e:  # noqa: BLE001
                print(f"Gemini vision error, retrying: {e}")
                time.sleep(1)
        return None

    def evaluate_image(self, image, eval_blur: bool = True, eval_partial: bool = False):
        """Same behavior as :meth:`util.chatGPT4.TextpromptGen.evaluate_image` using Gemini vision."""
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
