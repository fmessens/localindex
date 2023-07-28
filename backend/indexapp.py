from flask import (Flask, request, 
                   send_from_directory, 
                   jsonify)
from backend.indexfunctions import (index_new_files,
                                    QueryIndex,
                                    Summarizer)


def create_dbapp():
    app = Flask(__name__)
    Summ = Summarizer()

    @app.route('/indexnew')
    def indexnew():
        fileprogr = index_new_files()
        Qidx = QueryIndex()
        return fileprogr

    @app.route('/queryprocessed', methods=['GET'])
    def queryprocessed():
        Qidx = QueryIndex()
        qu = request.args.get('q', None)
        q_out = Qidx.processed_query(qu)
        texts, pagefiles, _ = q_out
        return jsonify({'texts': texts,
                        'pagefiles': pagefiles.to_dict('records')})
    
    @app.route('/querysummary', methods=['GET'])
    def querysummary():
        Qidx = QueryIndex()
        qu = request.args.get('q', None)
        q_out = Qidx.processed_query(qu)
        texts, pagefiles, _ = q_out
        texts_summ = [Summ.summarize(t) for t in texts]
        return jsonify({'texts': texts,
                        'texts_summ': texts_summ,
                        'pagefiles': pagefiles.to_dict('records')})
    
    @app.route('/query', methods=['GET'])
    def query():
        Qidx = QueryIndex()
        qu = request.args.get('q', None)
        samples, scores = Qidx.query(qu)
        print(samples.keys())
        return jsonify(samples)

    @app.route('/showPDFs')
    def send_pdf():
        pdf_path = request.args.get('path', None)
        print(pdf_path)
        pdf = pdf_path.split('/')[-1]
        path = pdf_path.replace(pdf, '')
        send = send_from_directory(path, pdf)
        print(send)
        return send

    return app
