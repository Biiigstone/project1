# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
#     USAGE     #
# python object_detection_benchmark.py -i <path-to-COCO-validation-images> -m <model> -p <protocol>


from __future__ import print_function
import argparse
import os
import time
import random
import requests
import numpy as np
from PIL import Image
import pandas as pd
import json
import tensorflow as tf

"""
IMAGES_PATH = '/home/jang/coco/val/val2017'
MODEL = 'rfcn'  # rfcn or ssd-mobilenet
PROTOCOL = 'grpc' # rest or grpc
BATCH_SIZE = 1
"""

NUM_ITERATION = 10
WARM_UP_ITERATION = 1
###################################디버깅용!!!!!11
##### 20220905 grpc input_tensor, inputs 수정됨
##### 20220907 num_detections 의미있도록 수정!


#FILE_NAME = "000000000139.jpg"  # File that takes long time to inference in local
#FILE_NAME = "000000000872.jpg"    # File that takes short time to inference in local
#IMAGE_FILE = "/home/jang/coco/val/val2017/"+FILE_NAME



#### default 설정 같음..
#Cloud_IP =  "34.22.67.170" 
Cloud_IP =  "203.237.143.28" 

Edge_IP = "192.168.1.35" 
#Host1_IP = "localhost"
#Host1_IP = Cloud_IP
Host1_IP = "localhost"
Host2_IP = "localhost"
#Delay_time_cloud = 0.05 #sec
#Delay_time_cloud = 0 #sec

def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise argparse.ArgumentTypeError("{} cannot be a link.".format(value))


def check_valid_folder(value):
    """Verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a directory.".
                                             format(value))
        check_for_link(value)
    return value


def check_valid_model(value):
    """Verifies model name is supported"""
    if value not in ('rfcn', 'ssd-mobilenet', 'test','CenterNet HourGlass104 Keypoints 512x512'):
        raise argparse.ArgumentError("Model name {} does not match 'rfcn' or 'ssd-mobilenet'.".
                                     format(value))
    return value


def check_valid_protocol(value):
    """Verifies protocol is supported"""
    if value not in ('rest', 'grpc'):
        raise argparse.ArgumentError("Protocol name {} does not match 'rest' or 'grpc'.".
                                     format(value))
    return value

def check_valid_host(value):
    """Verifies  is supported"""
    if value not in ('local', 'cloud', 'edge', 'cloud-edge'):
        raise argparse.ArgumentError("Host name {} does not match 'local' or 'cloud' or 'edge' or 'cloud-edge.".
                                     format(value))
    return value

def get_random_image(image_dir):
#    image_path = os.path.join(image_dir, random.choice(os.listdir(image_dir)))
#    image_path = os.path.join(image_dir, '/home/jang/coco/val/val2017/000000000285.jpg')
    image_path = os.path.join(image_dir, IMAGE_FILE)
    image = Image.open(image_path)
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
# 수정된 get_random_image는 랜덤 초이스 안합니다..
# 지정 이미지 파일 가져옵니다..

def make_request_cloud(batch_size):
    if PROTOCOL == 'rest':
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0).tolist(), batch_size, axis=0).tolist()
        return '{"instances" : %s}' % np_images
    elif PROTOCOL == 'grpc':
        import grpc
        import tensorflow as tf
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2_grpc
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0), batch_size, axis=0)
        options=[('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel(SERVER_URL1, options=options)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = MODEL
        request.model_spec.signature_name = 'serving_default'
        request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(np_images))
        return (stub, request)

def send_request_cloud(predict_request):
    global my_num_detections
    if PROTOCOL == 'rest':
        ladybug = requests.post(SERVER_URL1, data=predict_request)
        # num_detections의 값이 의미있도록 만들어주는 코드 스니펫. num_detections이 실제 표현될 객체의 수와 같은 값을 갖도록 합니다.
        print(ladybug)
        output_dict = ladybug.json()['predictions'][0]
        nod = np.array(output_dict['detection_scores'])
        my_num_detections = len(nod[nod>=0.5])
        print(ladybug)
        # print(my_num_detections) # visualize함수의 min_score_thresh와 동일하게 설정해주어야 합니다!
        # print('length : ', len(output_dict['detection_classes']))
        # print(ladybug.json())

    elif PROTOCOL == 'grpc':
        ladybug=predict_request[0].Predict(predict_request[1])
        output_dict = ladybug.outputs
 #       print(output_dict)
        nod = output_dict['detection_scores']
        nod = tf.make_ndarray(nod)
        my_num_detections = len(nod[nod >= 0.5])
        # print(f'my_num_detections : {my_num_detections}')
        # print(ladybug)


def make_request_edge(batch_size):
    if PROTOCOL == 'rest':
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0).tolist(), batch_size, axis=0).tolist()
        return '{"instances" : %s}' % np_images
    elif PROTOCOL == 'grpc':
        import grpc
        import tensorflow as tf
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2_grpc
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0), batch_size, axis=0)
  
        channel = grpc.insecure_channel(SERVER_URL2)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = MODEL
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(np_images))
        return (stub, request)

def send_request_edge(predict_request):
    if PROTOCOL == 'rest':
        requests.post(SERVER_URL2, data=predict_request)
    elif PROTOCOL == 'grpc':
        predict_request[0].Predict(predict_request[1])


def benchmark(batch_size=1, num_iteration=10, warm_up_iteration=1, num_cloud_load=10, delay_time=0):
    global latency, throughput, avg_time, avg_num_detections
    i = 0
    total_time = 0
    total_num_detections = 0
    avg_num_detections = 0
    for _ in range(num_iteration):
        i += 1
        if i <= num_cloud_load :
            print("Prepare for predict_request_cloud()")
            print()
            predict_request_cloud = make_request_cloud(batch_size)
        else :
            print("Prepare for predict_request_edge()")
            print()
            predict_request_edge = make_request_edge(batch_size)
        start_time = time.time() 
        if i <= num_cloud_load :
            print("Prepare for send_request_cloud()")
            send_request_cloud(predict_request_cloud) 
            inner_start_time = time.time()
            time.sleep(delay_time/1000) 
            inner_time_consum = time.time() - inner_start_time
            total_num_detections += my_num_detections
            print('Time to reach cloud : %.3f sec' % inner_time_consum)
        else :
            print("Prepare for send_request_edge()")
            send_request_edge(predict_request_edge)
        time_consume = time.time() - start_time

        print('Iteration %d: %.3f sec' % (i, time_consume))
        print()
        if i > warm_up_iteration:
            total_time += time_consume

    time_average = total_time / (num_iteration - warm_up_iteration)
    print('Average time: %.3f sec' % (time_average))
    avg_time = '%.3f' % (time_average)
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Latency: %.3f ms' % (time_average * 1000))
        latency = '%.3f' % (time_average * 1000)
    print('Throughput: %.3f images/sec' % (batch_size / time_average))
    throughput = '%.3f' % (batch_size / time_average)
    print(f'Average num_detections(.5) : {total_num_detections / num_iteration}')
    avg_num_detections = total_num_detections / num_iteration
    print(f'i = {i}')


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images_path", type=check_valid_folder, required=True,
                    help="Path to COCO validation directory")
    ap.add_argument("-m", "--model", type=check_valid_model, required=True,
                    help="Name of model (rfcn or ssd-mobilenet)")
    ap.add_argument("-p", "--protocol", type=check_valid_protocol, required=False, default="grpc",
                    help="Name of protocol (rest or grpc)")
    ap.add_argument("-b", "--batch_size", type=int, required=True,
                    help="Batch size")
    ap.add_argument("-a", "--host", type=check_valid_host, required=True,
                    help="Name of host (local or cloud or edge or cloud-edge)")
    ap.add_argument("-c", "--cloud_load", type=int, required=True,
                    help="Size of cloud load")
    ap.add_argument("-e", "--delay_time", type=int, required=True,
                    help="Delay time to Cloud")
    # 엑셀 파일이 저장될 위치를 받는 옵션 인자입니다.
    ap.add_argument("-r", "--results_path", type=check_valid_folder, required=True,
                    help="Path to save the result file")

    args = vars(ap.parse_args())

    IMAGES_PATH = args['images_path']
    MODEL = args['model']
    PROTOCOL = args['protocol']
    BATCH_SIZE = args['batch_size']
    Host_IP = args['host']
    Cloud_LOAD = args['cloud_load']
    DELAY_TIME = args['delay_time']
    RESULTS_PATH = args['results_path']


    ## 여기서 반복 시작하면 될듯..?
    img_file_list = os.listdir(IMAGES_PATH)
    img_file_list.sort()

    col_img_name = []
    col_img_size = []
    col_avg_time = []
    col_img_latency = []
    col_img_throughput = []
    col_num_detections = []

    for n in img_file_list:
        IMAGE_FILE = os.path.join(IMAGES_PATH, n)

        if Host_IP == 'local':
           Host1_IP = 'localhost'
        elif Host_IP == 'cloud':
            Host1_IP = Cloud_IP
        elif Host_IP == 'edge':
            Host1_IP = Edge_IP
        elif Host_IP == 'cloud-edge':
            Host1_IP = Cloud_IP
            Host2_IP = Edge_IP

        if PROTOCOL == 'rest':
    #        SERVER_URL = 'http://localhost:8501/v1/models/{}:predict'.format(MODEL)
    #        SERVER_URL = 'http://' + Host1_IP +':8501/v1/models/{}:predict'.format(MODEL)
            SERVER_URL1 = 'http://' + Host1_IP + ':8501/v1/models/{}:predict'.format(MODEL)
            SERVER_URL2 = 'http://' + Host2_IP + ':8501/v1/models/{}:predict'.format(MODEL)
    #        SERVER_URL = 'http://172.16.1.28:8501/v1/models/{}:predict'.format(MODEL)
        elif PROTOCOL == 'grpc':
    #        SERVER_URL = 'localhost:8500'
            SERVER_URL1 = Host1_IP + ':8500'
            SERVER_URL2 = Host2_IP + ':8500'
    #        SERVER_URL = '172.16.1.28:8500'    # Cloud (server)
    #        SERVER_URL = '172.16.1.24:8500'    # Edge (Laptop)

    #    print('\nSERVER_URL: {} \nIMAGES_PATH: {}'.format(SERVER_URL, IMAGES_PATH))
        print('\nSERVER_URL1: {} \nIMAGES_PATH: {}'.format(SERVER_URL1, IMAGES_PATH))
        print('\nSERVER_URL2: {} \nIMAGES_PATH: {}'.format(SERVER_URL2, IMAGES_PATH))
        print('\nWorkload between Cloud and Edge : {}-{}'.format(Cloud_LOAD, 10-Cloud_LOAD))

        print('\nStarting {} model benchmarking for latency on {}:'.format(MODEL.upper(), PROTOCOL.upper()))
        print('batch_size={}, num_iteration={}, warm_up_iteration={}, num_cloud_load={}, delay_time={}\n'.format(BATCH_SIZE, NUM_ITERATION, WARM_UP_ITERATION, Cloud_LOAD, DELAY_TIME/1000))
        print("Image File: %s" % IMAGE_FILE)
        print("File size : ", os.path.getsize(IMAGE_FILE), "bytes\n")
    #    benchmark(batch_size=BATCH_SIZE, num_iteration=20, warm_up_iteration=10)AttributeError: module 'tensorflow' has no attribute 'gfile'
        benchmark(batch_size=BATCH_SIZE, num_iteration=NUM_ITERATION, warm_up_iteration=WARM_UP_ITERATION, num_cloud_load=Cloud_LOAD, delay_time=DELAY_TIME)

        col_img_name.append(n)
        col_img_size.append(os.path.getsize(IMAGE_FILE))
        col_avg_time.append(avg_time)
        col_img_latency.append(latency)
        col_img_throughput.append(throughput)
        col_num_detections.append(avg_num_detections)

    df = pd.DataFrame(index = col_img_name)
    df['size'] = col_img_size
    df['avg_time'] = col_avg_time
    df['latency'] = col_img_latency
    df['throughput'] = col_img_throughput
    df['num_detections'] = col_num_detections
    # 파일 명 : 호스팅_모델명_프로토콜명.xlsx
    #
    # re = os.path.abspath(f'{RESULTS_PATH}/{Host_IP}_{MODEL}_{PROTOCOL}_{Cloud_LOAD}_{DELAY_TIME}.xlsx')
    re = os.path.abspath(f'{RESULTS_PATH}/{Host_IP}_{MODEL}_{PROTOCOL}.xlsx')
    df.to_excel(re)

    print()
    print(f'\nSaved in {RESULTS_PATH}')

# 20220808 path 수정, cloud, edge ip 수정, local 테스트 완료
# 20220811 cloud, cloud-edge 테스트 완료
# 20220815 옵션 인자 d 제거, 지정 디렉토리 내 모든 이미지로 iterate 하도록 수정
# v8 : test 모델들을 위해 check_valid_model()의 내부 리스트에 'test' 추가, [:100]
