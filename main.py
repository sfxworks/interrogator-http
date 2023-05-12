from flask import Flask, request
from PIL import Image
from io import BytesIO
from clip_interrogator import Config, Interrogator
import base64
import torch

app = Flask(__name__)

# Instantiate the Interrogator
config = Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k")
config.caption_model_name = 'blip-base'
config.caption_offload = True
config.clip_offload = True
config.chunk_size = 1024
config.flavor_intermediate_count = 1024
config.dtype = torch.float32  # Always use float32, regardless of device
ci = Interrogator(config)

@app.route('/interrogate', methods=['POST'])
def interrogate():
    image_data = base64.b64decode(request.json['image'])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    label = ci.interrogate(image)
    return {'label': label}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
