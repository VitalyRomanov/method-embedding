name := "GraphAnalysis"

version := "0.1"

scalaVersion := "2.11.12"

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
// https://mvnrepository.com/artifact/graphframes/graphframes

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.3",
  "org.apache.spark" %% "spark-sql" % "2.4.3"
)

// https://mvnrepository.com/artifact/org.apache.spark/spark-graphx
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.4.3"

libraryDependencies += "graphframes" % "graphframes" % "0.8.0-spark2.4-s_2.11"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
//libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
