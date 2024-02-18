import base64
from flask import Flask, request, jsonify, render_template, jsonify
import requests
from PIL import Image
import io

app = Flask(__name__)

# Replace with the actual URL of your image processing API endpoint
api_url = "https://backend.backgroundchanger.ai/api/v1/inpaint"

@app.route('/', methods=['GET'])
def display_ui():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get image and mask from form data
        image = request.files['image'].read()
        mask = request.files['mask'].read()

        # Determine image type
        image_type = Image.open(io.BytesIO(image)).format.lower()
        mask_type = Image.open(io.BytesIO(mask)).format.lower()

        # Encode image and mask to base64
        base64_image = base64.b64encode(image).decode('utf-8')
        base64_mask = base64.b64encode(mask).decode('utf-8')

        # Create payload dictionary with all parameters
        payload = {
                "image": f"data:image/{image_type};base64,{base64_image}",
                "mask": f"data:image/{mask_type};base64,{base64_mask}",
                "ldm_steps": 30,
                "ldm_sampler": "ddim",
                "zits_wireframe": True,
                "cv2_flag": "INPAINT_NS",
                "cv2_radius": 5,
                "hd_strategy": "Crop",
                "hd_strategy_crop_triger_size": 640,
                "hd_strategy_crop_margin": 128,
                "hd_trategy_resize_imit": 2048,
                "prompt": "",
                "negative_prompt": "out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature",
                "use_croper": False,
                "croper_x": 201,
                "croper_y": 384,
                "croper_height": 512,
                "croper_width": 512,
                "use_extender": False,
                "extender_x": 0,
                "extender_y": 0,
                "extender_height": 1280,
                "extender_width": 914,
                "sd_mask_blur": 12,
                "sd_strength": 1,
                "sd_steps": 50,
                "sd_guidance_scale": 7.5,
                "sd_sampler": "DPM++ 2M",
                "sd_seed": -1,
                "sd_match_histograms": False,
                "sd_freeu": False,
                "sd_freeu_config": {
                    "s1": 0.9,
                    "s2": 0.2,
                    "b1": 1.2,
                    "b2": 1.4
                },
                "sd_lcm_lora": False,
                "paint_by_example_example_image": None,
                "p2p_image_guidance_scale": 1.5,
                "enable_controlnet": False,
                "controlnet_conditioning_scale": 0.4,
                "controlnet_method": "",
                "powerpaint_task": "text-guided"
            }

        # Make POST request to the image processing API
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            # Process the response and save the result
            result_image = response.content

            # Return the result image as JSON
            return jsonify({'output_image': base64.b64encode(result_image).decode('utf-8')})
        else:
            return jsonify({'error': 'Error processing image'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
