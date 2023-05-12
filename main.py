from flask import Flask, request
from PIL import Image
from io import BytesIO
from clip_interrogator import Config, Interrogator
import base64

app = Flask(__name__)

# Instantiate the Interrogator
config = Config(clip_model_name="ViT-L-14/openai")
# config.apply_low_vram_defaults()
ci = Interrogator(config)


@app.route('/interrogate', methods=['POST'])
def interrogate():
    image_data = base64.b64decode(request.json['image'])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    label = ci.interrogate(image)
    return {'label': label}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
