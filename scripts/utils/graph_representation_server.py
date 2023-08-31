import http.server
import json
import os
import socketserver
import sys
from pathlib import Path

from SourceCodeTools.code.data.ast_graph.build_ast_graph import source_code_to_graph

PORT = 8000
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        server_path = str(Path(os.getcwd()).joinpath(Path(__file__)).parent.parent.parent.joinpath("res", "web_server").absolute())
        super().__init__(*args, directory=server_path, **kwargs)

    def do_POST(self):
        try:
            data_string = self.rfile.read(int(self.headers['Content-Length']))
            data = json.loads(data_string.decode("utf8"))

            map_variation = {
                "ast_graph": "v2.5",
                "ast_graph_with_instances": "v3.5",
                "cf_graph": "v3.5_control_flow"
            }
            graph = source_code_to_graph(data["source_code"], map_variation[data["variation"]])
            nodes = graph["nodes"]
            edges = graph["edges"]

            nodes = nodes.rename({"serialized_name": "name"}, axis=1)[["id", "type", "name"]]
            edges = edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1)[["id", "type", "src", "dst"]]

            json_responce = json.dumps({
                "nodes": nodes.to_dict(orient="records"),
                "edges": edges.to_dict(orient="records")
            })

            self.send_response(200)
            self.send_header('Content-type','application/json')
            self.end_headers()

            self.wfile.write(bytes(json_responce, "utf8"))
        except Exception as e:
            self.send_response(501)
            self.send_header('Content-type', 'application/json')
            self.end_headers()


#TODO
# requires documentation

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()