from flask import Flask, request, json, Response
import numpy as np

app = Flask(__name__)


@app.route('/similar_case', methods=['POST', 'GET'])
def get_similar_cases():
    if request.data:
        datas = json.loads(request.data)
        session_id = datas["session_id"]
        data = datas["data"]
        pageInfo = {"session_id":session_id, "pageInfo": "info"}
        return json.dumps({'data': pageInfo, "meta":{"code": 0, "msg":"请求成功！"}})
    else:
        return json.dumps({'data': None, "meta":{"code": 400, "msg":"参数错误"}})


@app.route('/downloadfile/', methods=['GET', 'POST'])
def downloadfile():
    if request.method == 'GET':
        fullfilename = request.args.get('filename')

        # fullfilename = '/root/allfile/123.txt'
        fullfilenamelist = fullfilename.split('/')
        filename = fullfilenamelist[-1]
        filepath = fullfilename.replace('/%s' % filename, '')


        # 普通下载
        # response = make_response(send_from_directory(filepath, filename, as_attachment=True))
        # response.headers["Content-Disposition"] = "attachment; filename={}".format(filepath.encode().decode('latin-1'))
        # return send_from_directory(filepath, filename, as_attachment=True)
        # 流式读取
        def send_file():
            store_path = fullfilename
            with open(store_path, 'rb') as targetfile:
                while 1:
                    data = targetfile.read(20 * 1024 * 1024)  # 每次读取20M
                    if not data:
                        break
                    yield data


        response = Response(send_file(), content_type='application/octet-stream')
        response.headers["Content-disposition"] = 'attachment; filename=%s' % filename  # 如果不加上这行代码，导致下图的问题
        return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)