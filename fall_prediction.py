import os
import time

from src.pipeline.fall_detect import FallDetector


def _fall_detect_config():

    _dir = os.path.dirname(os.path.abspath(__file__))
    _good_tflite_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        )
    _good_edgetpu_model = os.path.join(
        _dir,
        'ai_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        )
    _good_labels = 'ai_models/pose_labels.txt'
    config = {
        'model': {
            'tflite': _good_tflite_model,
            'edgetpu': _good_edgetpu_model,
            },
        'labels': _good_labels,
        'top_k': 3,
        'confidence_threshold': 0.6,
        'model_name':'mobilenet'
    }
    return config


def Fall_prediction(img_1,img_2,img_3=None):
    
    config = _fall_detect_config()
    result = None
    
    fall_detector = FallDetector(**config)

    def process_response(response):
        nonlocal result
        for res in response:
            result = res['inference_result']

    process_response(fall_detector.process_sample(image=img_1))
        
    time.sleep(fall_detector.min_time_between_frames)
    
    process_response(fall_detector.process_sample(image=img_2))
    
    if len(result) == 1:
        category = result[0]['label']
        confidence = result[0]['confidence']
        angle = result[0]['leaning_angle']
        keypoint_corr = result[0]['keypoint_corr']

        dict_res = {}
        dict_res["category"] = category
        dict_res["confidence"] = confidence
        dict_res["angle"] = angle
        dict_res["keypoint_corr"] = keypoint_corr
        return dict_res

    else:

        if img_3:
            
            time.sleep(fall_detector.min_time_between_frames)
            process_response(fall_detector.process_sample(image=img_3))

            if len(result) == 1:

                category = result[0]['label']
                confidence = result[0]['confidence']
                angle = result[0]['leaning_angle']
                keypoint_corr = result[0]['keypoint_corr']
                
                dict_res = {}
                dict_res["category"] = category
                dict_res["confidence"] = confidence
                dict_res["angle"] = angle
                dict_res["keypoint_corr"] = keypoint_corr
                return dict_res
        
    return None

if __name__ == '__main__':
    import socket
    import struct
    import base64
    import io
    from PIL import Image
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('192.168.137.1', 3257))
    server.listen(1)

    def receive_string(sck: socket.socket):
        try:
            # Receive the first 4 bytes that contain the length of the string
            int_bytes = sck.recv(4)
            string_len = struct.unpack('!I', int_bytes)[0]  # Network order
            
            # Now receive the string itself
            string_bytes = sck.recv(string_len)
            while len(string_bytes) < string_len:
                string_bytes += sck.recv(string_len - len(string_bytes))
            return string_bytes.decode('utf-8')
        except Exception as e:
            print(e)
            try:
                sck.close()
            except Exception as ignore:
                pass
            return None
        
    def string_to_bitmap(base64_encoded_string):
        try:
            # Decode the base64 string into bytes
            encode_bytes = base64.b64decode(base64_encoded_string)
            
            # Convert the bytes to an image
            image_data = io.BytesIO(encode_bytes)
            image = Image.open(image_data)
            return image
        except Exception as e:
            print(e)
            return None

    def handle_req(client_socket, client_address):
        data_string = receive_string(client_socket)
        print('received data from', client_address, ':', data_string)
        data_parts = data_string.split(':')
        event_type = data_parts[0]
        event_time = data_parts[1]
        base64_imgs = data_parts[2:]
        imgs = []
        for b64_img in base64_imgs: imgs.append(string_to_bitmap(b64_img))
        print('received', len(imgs), 'images')
        result = Fall_prediction(imgs[0], imgs[1], imgs[2])
        print('result:', result)

    while True:
        client_sock, client_addr = server.accept()
        handle_req(client_sock, client_addr)
        print('Connection from', client_addr)


