# ETB

## Getting Started

### Server (ETB Server + ElasticSearch + Kibana)

#### Install Docker
Install Docker supporting buildx.

```
$ sudo apt install docker.io -y
$ docker run -d --name registry -p 5000:5000 --restart always registry
```

Insecure-registries
```
$ sudo vi /etc/docker/daemon.json
```
If `daemon.json` does not exist, make an empty file and write the following line on it.
```
{ "insecure-registries":["server_ip:5000"] }
```
After that,
```
$ sudo service docker restart
```

#### Install Elasticsearch and Kibana

```
$ apt install openjdk-8-jre-headless
$ cd scripts
$ cd es
$ bash setup.sh
$ cd ../..
```

#### Install etb

Modify setup.py for server.

```
...
SERVER_FLAG = True
...
```

Install etb.
```
pip install -e .
```

#### Modify etb/config.py

##### Add node information.
```
	nodes = {
		`node_name`: {
			"ip":"192.168.0.11"
		}
	}
```
Change `node_name` to whatever you want.

#### Modify agent_access_tokens
```
	agent_access_token = {
		'node_token_n1': 'n1',
		...
	}
```
`agent_access_token` is needed to verify valid nodes.

### Edge task manager in an edge, a device/server doing inference

#### Install etb

Modify setup.py for edge.
```
...
SERVER_FLAG = False
...
``` 

Install etb.
```
pip install -e .
```

#### Install Docker
Install Docker supporting buildx.

```
$ sudo apt install docker.io -y
$ docker run -d --name registry -p 5000:5000 --restart always registry
```

Insecure-registries
```
$ sudo vi /etc/docker/daemon.json
```
If `daemon.json` does not exist, make an empty file and write the following line on it.
```
{ "insecure-registries":["server_ip:5000"] }
```
After that,
```
$ sudo service docker restart
```

#### Modify etb/config.py

##### Register Servcer Information
```
		log_server_ip = "server_ip"
		log_server_port = 9200
		etb_server = "http://server_ip:8080"
		docker_registry = "server_ip:5000"
```
Use your IP address instead of server_ip.


#### Modify edge agent related options
```
		access_token = "node_token_n1" # access_token for this node. (Refer to agent_access_tokens of the server's config.py)
		node_name = "n1" # this node name
		gpu = True # gpu usability
```

#### Execution
```
$ cd scripts/edge
$ python agent.py
```

If there is an error about ModuleNetFoundError for `etb, then execute the following command.
```
$ PYTHONPATH=/home/edge/nn-runtime/etb python agent.py
```

### Tester where the caller is located.

#### Install Docker
Install Docker supporting buildx.

```
$ sudo apt install docker.io -y
$ docker run -d --name registry -p 5000:5000 --restart always registry
```

Insecure-registries
```
$ sudo vi /etc/docker/daemon.json
```
If `daemon.json` does not exist, make an empty file and write the following line on it.
```
{ "insecure-registries":["server_ip:5000"] }
```
After that,
```
$ sudo service docker restart
```

#### Modify etb/config.py

##### Register Servcer Information
```
		log_server_ip = "server_ip"
		log_server_port = 9200
		etb_server = "http://server_ip:8080"
		docker_registry = "server_ip:5000"
```
Use your IP address instead of server_ip.


#### Modify user(tester) related options

It is not that important option. Do not change anything.
```
	access_token = "user_token" # user access token or node access token
```
