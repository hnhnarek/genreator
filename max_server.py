import subprocess
import os
import stat
import requests
import time
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import uuid
import base64

client_ip = "127.0.0.1"
client_port = 8282
SERVER_URL = "http://35.94.35.85:8989"
client = udp_client.SimpleUDPClient(client_ip, client_port)


# Define a handler for the OSC messages
def handler_function(address, *args):
    print(f"Received message {address} with arguments: {args}")
    print(args[0], args[0] == "generate")
    if args and args[0] == "generate":
        print('In')
        genres = args[1:len(args)//2+1]
        weights = [str(i) for i in args[len(args)//2+1:]]
 
        if len(genres) > 3:
            print("Too many genres")
        result = requests.get(f"{SERVER_URL}/generate", params={"genres": ','.join(genres), "weights": ','.join(weights)})
        if not result.ok:
            print("Error generating midi", result.content, result.status_code)
        else:
            print("Generated midi", result.json())
            res = result.json()['result']
            decoded_content = base64.b64decode(res)
            #result_file_name = f"{uuid.uuid4()}.mid"
            result_file_name = f"mix_{'_'.join([i.replace('/','-') for i in genres])}.mid"
            f_path = '/Users/ottavio/Desktop/midiGen/' + result_file_name #change according to your path
            with open(f_path,'wb') as f:
                f.write(decoded_content)
            res = client.send_message("read", str(f_path))
            print("Message sent",res)


# Create a dispatcher
dispatcher = dispatcher.Dispatcher()

# Map the handler function to the address "/test"
dispatcher.map("/receive", handler_function)

# Create a server, please remember to kill the server when you are done
server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 8989), dispatcher)

print("Serving on {}".format(server.server_address))
server.serve_forever()
