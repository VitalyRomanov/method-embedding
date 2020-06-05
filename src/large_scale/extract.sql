.headers on
.mode csv
.output edges.csv
SELECT * FROM edge;
.output nodes.csv
SELECT * FROM node;
.output element_component.csv
SELECT * FROM element_component;
.output source_location.csv
SELECT * FROM source_location;
.output occurrence.csv
SELECT * FROM occurrence;
.output filecontent.csv
SELECT * FROM filecontent;
.quit