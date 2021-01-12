import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.graphframes._
import org.apache.log4j.Logger
import org.apache.log4j.Level

//spark-shell --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11
//--conf "spark.driver.extraJavaOptions=-Djava.io.tmpdir=tmp"


object GraphAnalysis {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession.builder()
      .appName("GraphAnalysis").master("local[4]")
      .getOrCreate()

    val sc = SparkContext.getOrCreate()

    val nodes_path = args(0)
    val edges_path = args(1)
    val output_path = args(2)

    assert(nodes_path.endsWith(".parquet"), "Nodes should be stored in parquet format")
    assert(edges_path.endsWith(".parquet"), "Edges should be stored in parquet format")

    val nodes = spark.read.load(nodes_path)
    val edges = spark.read.load(edges_path)
      .withColumnRenamed("source_node_id", "src")
      .withColumnRenamed("target_node_id", "dst")
      .withColumnRenamed("id", "rel_id")
      .withColumnRenamed("type", "rel_type")

    println(s"Original graph")
    println(s"Nodes: ${nodes.count()}, Edges: ${edges.count()}")

    val g = GraphFrame(nodes, edges).dropIsolatedVertices()

    println(s"After dropping isolated")
    println(s"Nodes: ${g.vertices.count()}, Edges: ${g.edges.count()}")

    sc.setCheckpointDir("temp")
    val cc_result = g.connectedComponents.run()

    val component_count = cc_result.groupBy("component").count().sort(desc("count")).collect()

    println("Connected components")
    component_count.foreach(row => println(s"Component: ${row.getAs[Long]("component")}, count: ${row.getAs[Long]("count")}"))

    val largest_component = component_count(0).getAs[Long]("component")

    val g_cc = GraphFrame(cc_result, edges)

    val onlyConnected_g = g_cc
      .filterVertices(s"component == ${largest_component}")
      .dropIsolatedVertices()

    println(s"In the largest component")
    println(s"Nodes: ${onlyConnected_g.vertices.count()}, Edges: ${onlyConnected_g.edges.count()}")

    onlyConnected_g.vertices//.drop("component")
      .repartition(1)
      .write.save(s"${output_path}/component_0_nodes")

    onlyConnected_g.edges
      .repartition(1)
      .withColumnRenamed("rel_id", "id")
      .withColumnRenamed("rel_type", "type")
      .withColumnRenamed("src", "source_node_id")
      .withColumnRenamed("dst", "target_node_id")
      .write.save(s"${output_path}/component_0_edges")

//    write in degree counts
    val node_inDegrees = onlyConnected_g.vertices
      .join(onlyConnected_g.inDegrees, "id")

    node_inDegrees
      .sort(desc("inDegree"))//.select("serialized_name", "inDegree")
      .repartition(1)
      .write.format("csv").option("header", "true")
      .save(s"${output_path}/component_0_node_in_degrees")

//    write out degree counts
    val node_outDegrees = onlyConnected_g.vertices
      .join(onlyConnected_g.outDegrees, "id")

    node_outDegrees
      .sort(desc("outDegree"))//.select("serialized_name", "outDegree")
      .repartition(1)
      .write.format("csv").option("header", "true")
      .save(s"${output_path}/component_0_node_out_degrees")

    onlyConnected_g.edges.groupBy("rel_type").count()
      .withColumnRenamed("rel_type", "type")
      .sort(desc("count")).repartition(1)
      .write.format("csv").option("header", "true")
      .save(s"${output_path}/component_0_edge_type_count")

    onlyConnected_g.vertices.groupBy("type").count()
      .sort(desc("count")).repartition(1)
      .write.format("csv").option("header", "true")
      .save(s"${output_path}/component_0_node_type_count")

  }
}