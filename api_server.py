from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import json
import asyncio

from translator import Translator

class ApiRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = parse_qs(post_data)
        text = params.get('text', [''])[0]

        # print('post_data', post_data)
        # print('text', text)

        loop = asyncio.get_event_loop()
        translated_text = loop.run_until_complete(self.translate_text(post_data))

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'translated_text': translated_text}
        self.wfile.write(json.dumps(response).encode('utf-8'))

    async def translate_text(self, text):
        translator = Translator()
        translated_text = await translator.translate(text)
        return translated_text

class ApiServer:
    def __init__(self, host='localhost', port=3456):
        self.host = host
        self.port = port

    def start(self):
        server_address = (self.host, self.port)
        httpd = HTTPServer(server_address, ApiRequestHandler)
        print(f'Starting server on {self.host}:{self.port}...')
        httpd.serve_forever()

def main():
    server = ApiServer()
    server.start()

if __name__ == '__main__':
    main()
