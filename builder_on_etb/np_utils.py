import json, csv, threading, sys
from make_load import make_cpu_gpu_load, duration, load_functions

def print_with_overwrite(text):
    # 이전 출력 지우기
    sys.stdout.write('\r')  # 캐리지 리턴을 사용하여 커서를 줄의 맨 앞으로 이동
    sys.stdout.write(text + "                  \t")  # 새로운 텍스트 출력
    sys.stdout.flush() 

def load_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        dictionary = json.load(json_file)
    return dictionary

def save_dict_to_json(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
        
def csv_to_dict(file_path):
    # 결과를 저장할 리스트 초기화
    data_list = []

    # CSV 파일 열기
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        # DictReader 객체 생성
        csv_reader = csv.DictReader(csvfile)
        
        # 각 행을 딕셔너리로 변환하여 리스트에 추가
        for row in csv_reader:
            data_list.append(dict(row))
    
    return data_list

# CPU_GPU_MIN_MAX = [
#         {'cpu_min': 0, 'cpu_max': 25, 'gpu_min': 0, 'gpu_max': 25},
#         {'cpu_min': 25, 'cpu_max': 50, 'gpu_min': 0, 'gpu_max': 25},
#         {'cpu_min': 50, 'cpu_max': 75, 'gpu_min': 0, 'gpu_max': 25},
#         {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 0, 'gpu_max': 25},
#         {'cpu_min': 0, 'cpu_max': 25, 'gpu_min': 25, 'gpu_max': 50},
#         {'cpu_min': 25, 'cpu_max': 50, 'gpu_min': 25, 'gpu_max': 50},
#         {'cpu_min': 50, 'cpu_max': 75, 'gpu_min': 25, 'gpu_max': 50},
#         {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 25, 'gpu_max': 50},
#         {'cpu_min': 0, 'cpu_max': 25, 'gpu_min': 50, 'gpu_max': 75},
#         {'cpu_min': 25, 'cpu_max': 50, 'gpu_min': 50, 'gpu_max': 75},
#         {'cpu_min': 50, 'cpu_max': 75, 'gpu_min': 50, 'gpu_max': 75},
#         {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 50, 'gpu_max': 75},
#         {'cpu_min': 0, 'cpu_max': 25, 'gpu_min': 75, 'gpu_max': 100},
#         {'cpu_min': 25, 'cpu_max': 50, 'gpu_min': 75, 'gpu_max': 100},
#         {'cpu_min': 50, 'cpu_max': 75, 'gpu_min': 75, 'gpu_max': 100},
#         {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 75, 'gpu_max': 100},
#     ]

CPU_GPU_MIN_MAX = [
        {'cpu_min': 0, 'cpu_max': 75, 'gpu_min': 0, 'gpu_max': 75},
        {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 0, 'gpu_max': 75},
        {'cpu_min': 0, 'cpu_max': 75, 'gpu_min': 75, 'gpu_max': 100},
        {'cpu_min': 75, 'cpu_max': 100, 'gpu_min': 75, 'gpu_max': 100},
    ]


class LoadMaker(object):
    def __init__(self):
        self.thread = threading.Thread(target=make_cpu_gpu_load, args=())
        self.thread.daemon = True
        return
    
    def run(self):
        self.thread.start()
        return
    
    def cancel(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.cancel()

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()
    
    def test_duration_in_second(self):
        test_load_case_count = len(load_functions)
        return test_load_case_count * duration
    
    
    
