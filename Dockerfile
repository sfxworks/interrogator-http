FROM python:slim

# Install Python libraries
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install clip-interrogator==0.5.4
RUN pip3 install transformers==4.26.1

# Download models
RUN python3 -c "from clip_interrogator import Config, Interrogator; Interrogator(Config(clip_model_name='ViT-H-14/laion2b_s32b_b79k'))"

# Set up application
COPY . /app
WORKDIR /app

# Expose flask
EXPOSE 5000

# Set the command
CMD ["python3", "main.py"]

RUN pip3 install flask
