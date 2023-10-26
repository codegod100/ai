from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image as PILImage
from beam import App, Runtime, Image
import os

api_key = os.environ["HUGGINGFACE_TOKEN"]
app = App(
    name="fuyu",
    runtime=Runtime(
        gpu="A100",
        cpu=5,
        memory="64Gi",
        image=Image(
            python_packages=["Pillow", "transformers", "torch", "accelerate"],
            commands=[
                "pip install git+https://github.com/huggingface/transformers.git",
                "apt-get update",
                "apt-get install -y wget",
                "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin",
                "mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600",
                "wget -q https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb",
                "export DEBIAN_FRONTEND=noninteractive && dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb",
                "cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/",
            ],
        ),
    ),
)


def load_models():
    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id, token=api_key)
    model = FuyuForCausalLM.from_pretrained(
        model_id, device_map="cuda:0", token=api_key
    )

    return processor, model


@app.rest_api(loader=load_models)
def run(**inputs):
    # load model and processor
    processor, model = inputs["context"]
    # prepare inputs for the model
    text_prompt = "Generate a coco-style caption.\n"
    image_path = "https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png"
    image = PILImage.open(image_path)

    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to("cuda:0")

    # autoregressively generate text
    generation_output = model.generate(**inputs, max_new_tokens=7)
    generation_text = processor.batch_decode(
        generation_output[:, -7:], skip_special_tokens=True
    )
    # assert generation_text == ["A bus parked on the side of a road."]
    print(generation_text)
