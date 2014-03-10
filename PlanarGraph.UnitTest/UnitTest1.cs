﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PlanarGraph.Algorithm;
using PlanarGraph.Collections;
using PlanarGraph.Data;
using PlanarGraph.GF2;
using PlanarGraph.Parallel;

namespace PlanarGraph.UnitTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly IPlanarAlgorithm _gammaAlgorithm = new GammaAlgorithm();
        private readonly IPlanarAlgorithm _hopcroftTarjanAlgorithm = new HopcroftTarjanAlgorithm();

        [TestMethod]
        public void TestMethod1()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));
            Console.WriteLine(graph.ToString());
            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedAllGraphPaths =
                graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            Assert.AreEqual(1, subGraphs.Count());
            Assert.AreEqual(3, circles.Count());
            bool result1 = _gammaAlgorithm.IsPlanar(graph);
            bool result2 = _hopcroftTarjanAlgorithm.IsPlanar(graph);
            Console.WriteLine(@"{0}=={1}", result1, result2);
            Console.WriteLine();
            Assert.IsTrue(result1 == result2);
            Assert.IsTrue(result1);
            Assert.IsTrue(result2);
        }

        [TestMethod]
        public void TestVertexCollection()
        {
            var vertexCollection1 = new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6});
            var vertexCollection2 = new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5});
            var vertexCollection3 = new VertexUnsortedCollection(new List<int> {5, 1, 2, 3, 4});
            var vertexCollection4 = new VertexUnsortedCollection(new List<int> {5, 4, 3, 2, 1});
            var circle1 = new Circle(new VertexUnsortedCollection(new List<int> {2, 3, 4, 6, 1}));
            var circle2 = new Circle(new VertexUnsortedCollection(new List<int> {6, 4, 3, 2, 1}));
            var segment1 = new Segment(new VertexUnsortedCollection(new List<int> {6, 1}));
            var segment2 = new Segment(new VertexUnsortedCollection(new List<int> {6, 1}));
            var segmentCollection1 = new SegmentCollection {segment1};
            var segmentCollection2 = new SegmentCollection {segment2};
        }

        [TestMethod]
        public void TestRemoveAllTrees()
        {
            var graph = new Graph();
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 2})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 3})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 4})));
            graph.RemoveAllTrees();
            Assert.IsTrue(!graph.Vertices.Any());
            Assert.IsTrue(!graph.Any());
            var graph2 = new Graph();
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 3, 5, 2, 4})));
            Assert.AreEqual(5, graph2.Vertices.Count);
            Assert.AreEqual(10, graph2.Count);
        }

        [TestMethod]
        public void TestRemoveAllTrees1()
        {
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(@"Test #" + i);
                Graph graph = Graph.Random(20, 30);
                var graph2 = new Graph(graph);
                graph.RemoveAllTrees();
                graph2.RemoveAllTrees();
                graph2.RemoveAllTrees();
                Console.WriteLine(graph.ToString());
                Console.WriteLine(graph2.ToString());
                //Assert.IsTrue(graph.Vertices.Equals(graph2.Vertices));
                //Assert.IsTrue(graph.Segments.Equals(graph2.Segments));
                bool result = graph.Equals(graph2);
                Assert.IsTrue(result);
            }
        }

        [TestMethod]
        public void TestEquals()
        {
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(@"Test #" + i);
                Graph graph = Graph.Random(20, 30);
                var graph2 = new Graph(graph);
                bool result = graph.Equals(graph2);
                Assert.IsTrue(result);
            }
        }

        [TestMethod]
        public void TestRemoveAllBranches2()
        {
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(@"Test #" + i);
                Graph graph = Graph.Random(20, 30);
                var graph2 = new Graph(graph);
                graph.RemoveAllTrees();
                Console.WriteLine(graph.ToString());
                Console.WriteLine(graph2.ToString());
            }
        }

        [TestMethod]
        public void TestSplit()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 4, 6, 5})));
            var path = new Path();
            path.Add(new VertexUnsortedCollection(new List<int> {1, 6, 4, 5, 1}));
            IEnumerable<Path> paths = path.Split(graph);
            Console.WriteLine(string.Join(Environment.NewLine, paths.Select(item => item.ToString())));
            Assert.AreEqual(2, paths.Count());
        }

        [TestMethod]
        public void TestGammaAlgorithm()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));
            Assert.IsTrue(_gammaAlgorithm.IsPlanar(graph));
            var graph2 = new Graph();
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 3, 5, 2, 4})));
            Assert.IsFalse(_gammaAlgorithm.IsPlanar(graph2));
            var graph3 = new Graph();
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 4, 2, 6, 3, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 6})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {2, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {3, 4})));
            Assert.IsFalse(_gammaAlgorithm.IsPlanar(graph3));
        }

        [TestMethod]
        public void TestGammaAlgorithm1()
        {
            var graph = new Graph();
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {15, 6})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {2, 14})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {10, 4})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {14, 8})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {10, 9})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {2, 15})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {4, 7})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {4, 1})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {9, 3})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {13, 14})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {13, 3})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 13})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {5, 4})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {5, 10})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {12, 10})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {5, 13})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {0, 13})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {2, 1})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {8, 10})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {15, 9})));

            graph.RemoveIntermedians();
            Console.WriteLine(graph);

            var graph2 = new Graph(graph);
            graph2.RemoveAllTrees();
            Console.WriteLine(graph);
            Console.WriteLine(graph2);
//            Assert.IsTrue(graph.Equals(graph2));

            //graph.RemoveAllTrees();
            //Console.WriteLine(graph);

            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedAllGraphPaths =
                graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            bool result1 = _gammaAlgorithm.IsPlanar(new Graph(graph));
            bool result2 = _hopcroftTarjanAlgorithm.IsPlanar(new Graph(graph));
            Console.WriteLine(@"{0}=={1}", result1, result2);
            Console.WriteLine();
            Assert.IsTrue(result1 == result2);
        }

        [TestMethod]
        public void TestGammaAlgorithm2()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 3})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {2, 4})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 5, 6, 7})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 6})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {5, 7})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {6, 8, 9, 10})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {6, 9})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {8, 10})));
            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedAllGraphPaths =
                graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            bool result1 = _gammaAlgorithm.IsPlanar(graph);
            bool result2 = _hopcroftTarjanAlgorithm.IsPlanar(graph);
            Console.WriteLine(@"{0}=={1}", result1, result2);
            Console.WriteLine();
            Assert.IsTrue(result1 == result2);
            Assert.IsTrue(result1);
            Assert.IsTrue(result2);
        }

        [TestMethod]
        public void TestHopcroftTarjanAlgorithm()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));
            Assert.IsTrue(_hopcroftTarjanAlgorithm.IsPlanar(graph));
            var graph2 = new Graph();
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 3, 5, 2, 4})));
            Assert.IsFalse(_hopcroftTarjanAlgorithm.IsPlanar(graph2));
            var graph3 = new Graph();
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 4, 2, 6, 3, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 6})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {2, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {3, 4})));
            Assert.IsFalse(_hopcroftTarjanAlgorithm.IsPlanar(graph3));
        }

        [TestMethod]
        public void TestMinPaths()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));

            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedAllGraphPaths =
                graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            foreach (Circle circle in circles)
            {
                Console.WriteLine(circle);
                Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> minPaths = graph.GetMinPaths(circle,
                    cachedAllGraphPaths);
                Console.WriteLine(@"IsTauCircle = " + circle.IsTauCircle(graph, cachedAllGraphPaths));
                foreach (var pair in minPaths)
                {
                    Console.WriteLine("{0}=>{1}", pair.Key, string.Join(",", pair.Value.Select(item => item.ToString())));
                }
            }
            Assert.AreEqual(3, circles.Count());
        }

        [TestMethod]
        public void TestAlgorithms()
        {
            for (int i = 0; i < 20; i++)
            {
                Graph graph = Graph.Random(16, 20);
                Console.WriteLine(@"Test #" + i);
                Console.WriteLine(graph);
                bool result1 = _gammaAlgorithm.IsPlanar(new Graph(graph));
                bool result2 = _hopcroftTarjanAlgorithm.IsPlanar(new Graph(graph));
                Console.WriteLine(@"{0}=={1}", result1, result2);
                Console.WriteLine();
                Assert.IsTrue(result1 == result2);
            }
        }

        [TestMethod]
        public void TestCudafyMacLane()
        {
            var matrix =
                new BooleanMatrix(Enumerable.Range(0, 5).Select(i1 => Enumerable.Range(0, 10).Select(i => i%3 == 0)));
            int[] indexes = Enumerable.Range(1, 5).Select(i => i%5).ToArray();
            int rows = matrix.Count;
            int columns = matrix.Length;
            CudifyBooleanMatrixMacLane.SetBooleanMatrix(
                matrix.SelectMany(
                    row => Enumerable.Range(0, columns).Select(i => (i < row.Count || row[i]) ? 1 : 0))
                    .ToArray(), rows, columns);
            CudifyBooleanMatrixMacLane.SetIndexes(indexes.ToArray());
            CudifyBooleanMatrixMacLane.Execute();
            int macLane = CudifyBooleanMatrixMacLane.GetMacLane();
            Console.WriteLine(macLane);
            Assert.AreEqual(matrix.MacLane, macLane);
        }
    }
}