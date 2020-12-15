import sys
import pandas as pd
from neo4j import GraphDatabase
import argparse

from SourceCodeTools.data.sourcetrail.Dataset import load_data



def read_type_map(path):
    """
    Reads types into a dictionary. Types are stored as scv in the format (int_type, description)
    :param path:
    :return:
    """
    return dict(pd.read_csv(path).values)


def main(args):
    node_map = read_type_map(args.sourcetrail_nodes)
    edge_map = read_type_map(args.sourcetrail_edges)
    edge_map.update(read_type_map(args.ast_edges))

    nodes, edges = load_data(args.node_path, args.edge_path)
    nodes['type'] = nodes['type'].map(lambda x: node_map[x])
    edges['type'] = edges['type'].map(lambda x: edge_map[x])

    print()

    uri = f"neo4j://{args.host}:{args.port}"
    driver = GraphDatabase.driver(uri, auth=(f"{args.user}", f"{args.password}"))

    def create_node(tx, id_, name_, type_):
        tx.run(f"CREATE (n:Node{{id:{id_}, name:'{name_}', type:'{type_}'}})")

    def create_edge(tx, src, dst, type):
        tx.run(f"""
        match (s:Node{{id:{src}}}), (d:Node{{id:{dst}}})
        merge (s)-[:{type}]->(d)
        """)

    def create_indexes(tx):
        tx.run("create index on :Node(id)")
        tx.run("create index on :Node(name)")
        tx.run("create index on :Node(type)")


    print("Importing nodes")
    # session = driver.session().__enter__()
    with driver.session() as session:
        # https://neo4j.com/docs/api/python-driver/current/api.html#explicit-transactions

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
        create_indexes(tx)
        tx.commit()
        tx.close()
        tx = session.begin_transaction()

        print("Importing edges")

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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--host', dest='host', default="localhost",
                        help='')
    parser.add_argument('--port', "-P", dest='port', default="7687",
                        help='')
    parser.add_argument('--user', "-u", dest='user', default="neo4j",
                        help='')
    parser.add_argument('--password', "-p", dest='password', default=None,
                        help='')
    parser.add_argument('--node_path', "-n", dest='node_path', default=None,
                        help='')
    parser.add_argument('--edge_path', "-e", dest='edge_path', default=None,
                        help='')
    parser.add_argument('--sourcetrail_edges', "-se", dest='sourcetrail_edges', default=None,
                        help='')
    parser.add_argument('--sourcetrail_nodes', "-sn", dest='sourcetrail_nodes', default=None,
                        help='')
    parser.add_argument('--ast_edges', "-ae", dest='ast_edges', default=None,
                        help='')
    parser.add_argument('--batch_size', "-b", dest='batch_size', default=10000, type=int,
                        help='')

    main(parser.parse_args())



