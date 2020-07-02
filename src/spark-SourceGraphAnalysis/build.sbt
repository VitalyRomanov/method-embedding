name := "src"

version := "0.1"

scalaVersion := "2.11.12"

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
// https://mvnrepository.com/artifact/graphframes/graphframes

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.0",
  "org.apache.spark" %% "spark-sql" % "2.1.0"
)

libraryDependencies += "graphframes" % "graphframes" % "0.6.0-spark2.3-s_2.11"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
