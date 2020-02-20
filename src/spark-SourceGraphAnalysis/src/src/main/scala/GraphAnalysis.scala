import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.graphframes._

//spark-shell --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11



object GraphAnalysis {

  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .appName("GraphAnalysis")
      .getOrCreate()

    val lang = "python"

    var nodes_path = ""
    var edges_path = ""

    if (lang == "python"){
      nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv"
      edges_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/non-ambiguous_edges.csv"
    } else if (lang == "java") {
      nodes_path = "/Volumes/External/datasets/Code/source-graphs/java-source-graph-v2/normalized_sourcetrail_nodes.csv"
      edges_path = "/Volumes/External/datasets/Code/source-graphs/java-source-graph-v2/edge.csv"
    }

    val nodes = spark.read.format("csv").option("header",true).load(nodes_path)
    val edges = spark.read.format("csv").option("header",true).load(edges_path).withColumnRenamed("source_node_id", "src").withColumnRenamed("target_node_id", "dst").withColumnRenamed("id", "rel_id").withColumnRenamed("type", "rel_type")

    val g = GraphFrame(nodes, edges)

//    sc.setCheckpointDir("temp")
    val cc_result = g.connectedComponents.run()

    var onlyConnected_g : GraphFrame = null

    if (lang == "python") {
      onlyConnected_g = g.filterVertices("component == \"0\"").filterEdges("rel_type != \"4\"").dropIsolatedVertices()
    } else if (lang == "java") {
      onlyConnected_g = g.filterVertices("component == \"0\"").dropIsolatedVertices()
    }

    onlyConnected_g.vertices.write.format("csv").save("components_0_nodes.csv")
    onlyConnected_g.edges.write.format("csv").save("components_0_edges.csv")

    val node_degres = nodes.join(g.inDegrees, "id")

    node_degres.sort(desc("inDegree")).write.format("csv").save("node_degrees.csv")

    node_degres.groupBy("type").agg(mean("inDegree")).sort(desc("avg(inDegree)")).write.format("csv").save("node_avg_indegree.csv")

    val node_outDegres = nodes.join(g.outDegrees, "id")

    node_outDegres.sort(desc("outDegree")).write.format("csv").save("node_out_degrees.csv")

    node_outDegres.groupBy("type").agg(mean("outDegree")).sort(desc("avg(outDegree)")).write.format("csv").save("node_avg_out_indegree.csv")

    edges.groupBy("rel_type").count().write.format("csv").save("edge_type_count.csv")

    x

    val cc_result = g.connectedComponents.run()

    cc_result.write.format("csv").save("components.csv")

    cc_result.groupBy("component").count().orderBy().write.format("csv").save("components_count.csv")



//    val pr_results = g.pageRank.resetProbability(0.15).tol(0.01).run()

//    pr_results.vertices.write.format("csv").save("pagerank_vertices.csv")

//    pr_results.edges.write.format("csv").save("pagerank_edges.csv")

//    val textual_edges_path = "/Volumes/Untitled/normalized_sourcetrail_edges.csv"
//    val textual_edges = spark.read.format("csv").option("header",true).load(textual_edges_path).withColumnRenamed("source_node_id", "src").withColumnRenamed("target_node_id", "dst").withColumnRenamed("id", "rel_id").withColumnRenamed("type", "rel_type")
//    textual_edges.filter("src == 'twisted.internet._glibbase.GlibReactorBase.__init__'").count

  }
}