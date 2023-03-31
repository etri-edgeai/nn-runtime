def set_config(mode):

    ############ Edge Pool Setup for Server ##########################################
    if mode == "server":

        global nodes, node_names, user_access_tokens, agent_access_tokens, upload_folder 
        nodes = {
           'n1': {
               "ip": "129.254.165.171" # 엣지 노드 ip 설정
           },
           'n2': {
               "ip": "192.168.0.12" # 엣지 노드 ip 설정
            },
           'n3': {
               "ip": "192.168.0.13" # 엣지 노드 ip 설정
           },
           'n4': {
               "ip": "192.168.0.10" # 엣지 노드 ip 설정
            
            }
        } 
        # 참고로, 현재는 노드 이름만이 의미가 있다. 나머지는 부가 설명 정보다.

        node_names = []
        for n in nodes:
           node_names.append(n)

        # 사용자 access_tokens 정보를 입력한다.
        # 테스터는 이 토큰값을 알아야 서버에 요청을 보낼 수 있다.
        user_access_tokens = {
           'user_token': 'user_id'
        }

        # 에이전트 access_tokens 정보를 입력한다.
        # 에이전트는 이 토큰값을 알아야 서버에 요청을 보낼 수 있다.
        agent_access_tokens = {
           'node_token_n1': 'n1',
           'node_token_n2': 'n2',
           'node_token_n3': 'n3',
           'node_token_n4': 'n4',
        }
        #################################################################################

        upload_folder = "./"

    else:
        global log_server_ip, log_server_port, etb_server, docker_registry, access_token

        # Common
        log_server_ip = "129.254.165.171"
        log_server_port = 9200
        etb_server = "http://129.254.165.171:8080"
        docker_registry = "129.254.165.171:5000"

        if mode == "usr": # Caller to ETB
            access_token = "user_token" # user access token or node access token

        else: # Edge Agent
            global node_name, gpu
            access_token = "node_token_n1" # user access token or node access token
            node_name = "n1" # used for logging in es
            gpu = True # gpu usability
