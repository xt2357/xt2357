import json

import cherrypy
import jieba
import lstm_with_tag_relation as rank_model
import refined_models as tree_model

DEFAULT_ENCODING = 'utf-8'

def to_json(content):
    cherrypy.response.headers['Content-Type'] = 'application/json'
    cherrypy.response.headers['Content-Encoding'] = DEFAULT_ENCODING
    return json.dumps(content, ensure_ascii=False).encode(DEFAULT_ENCODING)

class CDCTaggerService(object):
    def __init__(self):
        rank_model.load_classify_model()
        tree_model.load_classify_model()

    @cherrypy.expose
    def index(self):
        return "Hello world!"

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def tag(self):
        res = {'tags': []}
        json_data = cherrypy.request.json
        content = json_data['content']
        classifier = json_data['classifier']
        if classifier == u'ranking_model':
            for tag, confidence in rank_model.classify(content):
                res['tags'].append({tag: unicode(confidence)})

        else:
            for tag in tree_model.classify(content):
                res['tags'].append({tag: u'NAN'})
        return res

if __name__ == '__main__':
    cherrypy.config.update({
        'server.socket_host': '0.0.0.0',
        'server.socket_port': 8083,
    })
    cherrypy.quickstart(CDCTaggerService())
