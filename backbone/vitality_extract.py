"""Extract vitality layers for FLUX.2 klein"""
import torch
from diffusers import Flux2KleinPipeline
from transformers import AutoImageProcessor, AutoModel
import getpass
import os

device = "cuda"
dtype = torch.bfloat16
k = 64

# Do not use distilled model
# pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

# No of blocks: 25
# 5 double, 20 single per pass
# print(len(pipe.transformer.transformer_blocks))
# print(len(pipe.transformer.single_transformer_blocks))

# For k = 64
# We generate k seeds
def generate_seeds(k):
    seeds = list(torch.randint(high=100000, size=(k,), device=device, generator=torch.Generator(device=device).manual_seed(42)))
    return seeds

def generate_text_prompts(k=k, out_file="prompts.txt"):
    """Generate text prompts and writes them to a file."""
    # Gemini setup
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    from langchain_google_genai import ChatGoogleGenerativeAI

    # Initialize model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    scene_text = """
    You are a helpful assistant that generates scenery from different kinds of worlds. It may be a real-world scene or from a fictional genre.
    Generate a descriptive prompt that could be fed into a text-to-image model.
    The prompt should enable the model to generate an expressive image in a variety of styles
    CRITICAL: The prompt should also enforce the model to generate images with clear foreground objects, background scenery, and sky.
    Return the prompt directly.
    """

    # Generate k text prompts
    with open(out_file, "w") as f:
        for _ in range(k):
            resp = llm.invoke(scene_text)
            print(resp.text)
            f.write(resp.text + "\n")

def read_text_prompts(file="prompts.txt"):
    with open(file, "r") as f:
        return f.readlines()

# For each of the k prompts by Gemini, fixed seed i
# Pass through model with all layers, fixed seed i, encode DinoV2
# Pass through model with one layer deactivated i, encode DinoV2
# Total: 64 * (25 + 1) iterations
def calc_perceptual_similarity(prompts, seeds, out_file="sim.csv", save=True):
    import csv
    
    # Init FLUX
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

    # Obtain no of layers to ablate
    n_layers = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
    
    # Init DINOv2
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dinov2 = AutoModel.from_pretrained('facebook/dinov2-base')

    def predict_dinov2(image):
        inputs = processor(images=image, return_tensors="pt")
        outputs = dinov2(**inputs)
        last_hidden_states = outputs.last_hidden_state
        if last_hidden_states.dim() > 3:
            last_hidden_states = last_hidden_states.view(last_hidden_states.size(0), last_hidden_states.size(1), -1)
        return last_hidden_states
    
    def predict_flux2(ind, save="outputs", skipped_layer=None):
        mm_skip_blocks = None
        single_skip_blocks = None
        if skipped_layer is not None:
            if skipped_layer < len(pipe.transformer.transformer_blocks):
                mm_skip_blocks = [skipped_layer]
            else:
                single_skip_blocks = [skipped_layer - len(pipe.transformer.transformer_blocks)]

        image = pipe(
            prompt=prompts[ind],
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator(device=device).manual_seed(int(seeds[ind])),
            mm_skip_blocks=mm_skip_blocks,
            single_skip_blocks=single_skip_blocks
        ).images[0]
        
        # Creates save directory
        save_dir = save + f"/prompt_{ind}"
        os.makedirs(save_dir, exist_ok=True)
        save_file = save_dir + f"/flux_p{ind}"
        if skipped_layer is not None:
            save_file += f"_l{skipped_layer}"

        image.save(save_file + ".png")
        return image

    assert len(prompts) == len(seeds)
    k = len(prompts)

    if save and not os.path.exists(out_file):
        with open(rf"{out_file}", "a") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_idx"] + [str(x) for x in range(n_layers)])

    for i in range(k):
        print(f"Prompt {i + 1} of {k}")
        # Pass through model with all layers, fixed seed l, encode DinoV2
        baseline = predict_flux2(i)
        baseline_emb = predict_dinov2(baseline)

        # For each of the layers, we pass through it once
        perc_sims = []
        for l in range(n_layers):
            # Pass through model with one layer deactivated l, encode DinoV2
            img = predict_flux2(i, skipped_layer=l)
            emb = predict_dinov2(img)

            # Calculate cosine similarity
            perc_sim = torch.nn.functional.cosine_similarity(baseline_emb.detach(), emb.detach(), dim=-1).mean()
            print(f"Perceptual similarity skipping layer {l}: {float(perc_sim):.4f}")
            perc_sims.append(float(perc_sim))

        # Append perceptual similarities for that prompt
        if save:
            with open(rf"{out_file}", "a") as f:
                writer = csv.writer(f)
                writer.writerow([str(i)] + perc_sims)

def calc_vitality(file="sim.csv", out_file="vitality.csv", plot=True):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file)

    # Convert to 2D numpy array, remove first column (prompt idx)
    a = df.to_numpy()[:, 1:]

    # Take mean and variance of each column
    sims = a.mean(axis=0)
    vitalities = 1 - sims
    var = a.var(axis=0)

    # Save to csv
    df_v = pd.DataFrame(
        data=np.vstack((sims, vitalities, var)).T,
        index=np.arange(len(sims)),
        columns=["similarity", "vitality", "variance"]
    )
    df_v.to_csv(out_file)

    # Plot similarity scores if plot=True
    if plot:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.set_title("Perceptual similarity against layer dropped")
        ax.set_xlabel("Layer dropped")
        ax.set_ylabel("Perceptual similarity (mean)")
        ax.scatter(
            x=np.arange(len(sims)),
            y=sims
        )
        for l, s in enumerate(sims):
            plt.text(l, s, str(l))

        f.savefig("sim_plot.png")


if __name__ == "__main__":
    # prompts = read_text_prompts("prompts.txt")
    # seeds = generate_seeds(k=k)
    # calc_perceptual_similarity(
    #     prompts=prompts, seeds=seeds, out_file="sim.csv", save=True
    # )
    calc_vitality()
    MM_VITAL_BLOCKS = [0, 4]
    SINGLE_VITAL_BLOCKS = [9, 10, 15, 24]
