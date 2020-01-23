LOAD CSV WITH HEADERS FROM 'file:///python_data/normalized_sourcetrail_nodes.csv' as row FIELDTERMINATOR ','
CREATE (:Object {id:toInteger(row.id), type:toInteger(row.type), name:row.serialized_name})

CREATE INDEX ON :Object(id)
CREATE INDEX ON :Object(type)
CREATE INDEX ON :Object(name)

LOAD CSV WITH HEADERS FROM 'file:///python_data/edges.csv' as row FIELDTERMINATOR ','
MATCH (s:Object{id:toInteger(row.source_node_id)})
MATCH (d:Object{id:toInteger(row.target_node_id)})
MERGE (s)-[:REL {id: toInteger(row.id), type: toInteger(row.type)}]->(d)

