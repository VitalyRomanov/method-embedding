import pandas as pd
from neo4j import GraphDatabase
import argparse

from SourceCodeTools.code.data.dataset.Dataset import load_data
# from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types, edge_types


def read_type_map(path):
    """
    Reads types into a dictionary. Types are stored as scv in the format (int_type, description)
    :param path:
    :return:
    """
    return dict(pd.read_csv(path).values)


def main(args):

    nodes, edges = load_data(args.node_path, args.edge_path)

    print()

    uri = f"neo4j://{args.host}:{args.port}"
    driver = GraphDatabase.driver(uri, auth=(f"{args.user}", f"{args.password}"))

    def create_node(tx, id_, name_, node_type):
        tx.run(f"CREATE (n:{node_type.replace('#','_')}{{id:{id_}, name:'{name_}'}})")

    def create_edge(tx, src, dst, edge_type):
        tx.run(f"""
        match (s{{id:{src}}}), (d{{id:{dst}}})
        merge (s)-[:{edge_type}]->(d)
        """)

    def create_indexes(tx, types):
        for type in types:
            tx.run(f"create index on :{type.replace('#','_')}(id)")
            tx.run(f"create index on :{type.replace('#','_')}(name)")

    with driver.session() as session:
        # https://neo4j.com/docs/api/python-driver/current/api.html#explicit-transactions

        print("Importing nodes")

        tx = session.begin_transaction()
        for ind, row in nodes.iterrows():
            create_node(tx, row['id'], row['name'], row['type'])
            if (ind + 1) % args.batch_size == 0:
                print(f"{ind}/{len(nodes)}", end="\r")
                tx.commit()
                tx.close()
                tx = session.begin_transaction()
        tx.commit()
        tx.close()

        print()

        print("Creating indexes")
        tx = session.begin_transaction()
        create_indexes(tx, types=nodes['type'].unique().tolist())
        tx.commit()
        tx.close()

        print("Importing edges")

        tx = session.begin_transaction()
        for ind, row in edges.iterrows():
            create_edge(tx, row['src'], row['dst'], row['type'])
            if (ind + 1) % args.batch_size == 0:
                print(f"{ind}/{len(edges)}", end="\r")
                tx.commit()
                tx.close()
                tx = session.begin_transaction()
        tx.commit()
        tx.close()

        print()

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--host', dest='host', default="localhost",
                        help='')
    parser.add_argument('--port', "-P", dest='port', default="7687",
                        help='')
    parser.add_argument('--user', "-u", dest='user', default="neo4j",
                        help='')
    parser.add_argument('--password', "-p", dest='password', default=None,
                        help='')
    parser.add_argument('node_path',
                        help='')
    parser.add_argument('edge_path',
                        help='')
    parser.add_argument('--batch_size', "-b", dest='batch_size', default=1000, type=int,
                        help='')

    main(parser.parse_args())
