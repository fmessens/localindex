from flask import (Flask, request, 
                   send_from_directory, 
                   jsonify)
from backend.indexfunctions import (index_new_files,
                                    QueryIndex,
                                    Summarizer)


def create_dbapp():
    """Create a backend Flask application.

    Returns:
        Flask.app: Flask application
    """
    app = Flask(__name__)
    Summ = Summarizer()

    @app.route('/indexnew')
    def indexnew():
        """Index new files.

        Returns:
            dict: status of indexing
        """
        fileprogr = index_new_files()
        return fileprogr

    @app.route('/queryprocessed', methods=['GET'])
    def queryprocessed():
        """Query index and preprocess results.

        Returns:
            dict: jsonified results of texts, files and pages
        """
        Qidx = QueryIndex()
        qu = request.args.get('q', None)
        q_out = Qidx.processed_query(qu)
        texts, pagefiles, _ = q_out
        return jsonify({'texts': texts,
                        'pagefiles': pagefiles.to_dict('records')})
    
    @app.route('/querysummary', methods=['GET'])
    def querysummary():
        """Query index, preprocess and summarize results.

        Returns:
            dict: jsonified results of texts, text summaries, files and pages
        """
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
        """Query the index.

        Returns:
            dict: relevant samples
        """
        Qidx = QueryIndex()
        qu = request.args.get('q', None)
        samples, scores = Qidx.query(qu)
        print(samples.keys())
        return jsonify(samples)

    @app.route('/showPDFs')
    def send_pdf():
        """Encode and send the pdf file of the path 
            over the request response.

        Returns:
            dict: response data
        """
        pdf_path = request.args.get('path', None)
        pdf = pdf_path.split('/')[-1]
        path = pdf_path.replace(pdf, '')
        send = send_from_directory(path, pdf)
        return send

    return app
