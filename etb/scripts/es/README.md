# Elastic Search

ETB works with Elasticsearch and Kibana 7.14.0.
This directory is made for handling them.

```
bash setup.sh
```

To run Elasticsearch,

```
bash exec_es.sh
```
After running it, you can find the process id from `elasticsearch.pid`.

To run Kibana,

```
bash exec_kibana.sh
```
The process id will be in `kibana.pid`.

Such PID information will be needed for the case that ETB fails with some reasons.
