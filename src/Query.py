class Query:
    def __init__(self, query_id, query_name, results=None):
        self.query_id = query_id
        self.query_name = query_name
        self.results = results if results is not None else []

    def add_result(self, result):
        self.results = result

    def get_results(self):
        return self.results

    def __repr__(self):
        return f"Query(query_id={self.query_id}, query_name='{self.query_name}', results={self.results})"