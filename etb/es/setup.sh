
echo 'download and tar... es'
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.14.0-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.14.0-linux-x86_64.tar.gz

echo 'download and tar... kibana'
wget https://artifacts.elastic.co/downloads/kibana/kibana-7.14.0-linux-x86_64.tar.gz
tar -xzf kibana-7.14.0-linux-x86_64.tar.gz


echo 'kibana setup'
sed -i 's/#server.host: "localhost"/server.host: "0.0.0.0"/g' ./kibana-7.14.0-linux-x86_64/config/kibana.yml

echo 'setup max_map_count'
sudo sysctl -w vm.max_map_count=262144

echo '(manually) vi elasticsearch-*/config/elasticsearch.yml'
echo 'network.host: 0.0.0.0'
echo 'discovery.type: single-node'
