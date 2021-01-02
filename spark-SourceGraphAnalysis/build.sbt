name := "GraphAnalysis"

version := "0.1"

scalaVersion := "2.12.10"

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
// https://mvnrepository.com/artifact/graphframes/graphframes

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.1",
  "org.apache.spark" %% "spark-sql" % "3.0.1"
)

// https://mvnrepository.com/artifact/org.apache.spark/spark-graphx
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.0.1"

libraryDependencies += "graphframes" % "graphframes" % "0.8.1-spark3.0-s_2.12"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
//libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
