using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using PlanarGraph.Algorithm;
using PlanarGraph.Array;
using PlanarGraph.Collections;
using PlanarGraph.Data;
using PlanarGraph.GF2;
using PlanarGraph.Parallel;

namespace PlanarGraph.UnitTest
{
    [TestClass]
    public class UnitTest1
    {
        private readonly IPlanarAlgorithm _gammaAlgorithm = new GammaAlgorithm
        {
            WorkerLog = WorkerLog,
        };

        private readonly IPlanarAlgorithm _macLaneAlgorithm = new MacLaneAlgorithm
        {
            WorkerLog = WorkerLog,
        };

        private static void WorkerLog(string text)
        {
            Console.WriteLine(text);
        }

        [TestMethod]
        public void TestMethod1()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));
            Console.WriteLine(graph.ToString());
            Dictionary<int, PathDictionary> cachedAllGraphPaths =
                graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            Assert.AreEqual(1, subGraphs.Count());
            Assert.AreEqual(3, circles.Count());
            bool result1 = _gammaAlgorithm.IsPlanar(graph);
            bool result2 = _macLaneAlgorithm.IsPlanar(graph);
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
        public void TestEquals1()
        {
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(@"Test #" + i);
                var segment1 = new Segment(new VertexUnsortedCollection(new StackListQueue<int> {i, i + 1}));
                var segment2 = new Segment(new VertexUnsortedCollection(new StackListQueue<int> {i + 1, i}));
                var segment3 = new Segment(new VertexUnsortedCollection(new StackListQueue<int> {i + 1, i + 2}));
                Assert.IsTrue(segment1.Equals(segment2));
                Assert.IsFalse(segment1.Equals(segment3));
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
            IEnumerable<Path> paths = path.SplitBy(graph);
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

            Dictionary<int, PathDictionary> cachedAllGraphPaths = graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            bool result1 = _gammaAlgorithm.IsPlanar(new Graph(graph));
            bool result2 = _macLaneAlgorithm.IsPlanar(new Graph(graph));
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
            Dictionary<int, PathDictionary> cachedAllGraphPaths = graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            IEnumerable<Graph> subGraphs = graph.GetAllSubGraphs();
            Console.WriteLine(graph.ToString());
            Console.WriteLine(string.Join(Environment.NewLine, subGraphs.Select(item => item.ToString())));
            Console.WriteLine(string.Join(Environment.NewLine, circles.Select(item => item.ToString())));
            bool result1 = _gammaAlgorithm.IsPlanar(graph);
            bool result2 = _macLaneAlgorithm.IsPlanar(graph);
            Console.WriteLine(@"{0}=={1}", result1, result2);
            Console.WriteLine();
            Assert.IsTrue(result1 == result2);
            Assert.IsTrue(result1);
            Assert.IsTrue(result2);
        }

        [TestMethod]
        public void TestMacLaneAlgorithm()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));
            Assert.IsTrue(_macLaneAlgorithm.IsPlanar(graph));
            var graph2 = new Graph();
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph2.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 3, 5, 2, 4})));
            Assert.IsFalse(_macLaneAlgorithm.IsPlanar(graph2));
            var graph3 = new Graph();
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 4, 2, 6, 3, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 6})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {2, 5})));
            graph3.Add(new Circle(new VertexUnsortedCollection(new List<int> {3, 4})));
            Assert.IsFalse(_macLaneAlgorithm.IsPlanar(graph3));
        }

        [TestMethod]
        public void TestMinPaths()
        {
            var graph = new Graph();
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 6})));
            graph.Add(new Circle(new VertexUnsortedCollection(new List<int> {1, 2, 3, 4, 5})));
            graph.Add(new Segment(new VertexUnsortedCollection(new List<int> {1, 7})));

            Dictionary<int, PathDictionary> cachedAllGraphPaths = graph.GetAllGraphPaths();
            IEnumerable<Circle> circles = graph.GetAllGraphCircles(cachedAllGraphPaths);
            foreach (Circle circle in circles)
            {
                Console.WriteLine(circle);
                Dictionary<KeyValuePair<Vertex, Vertex>, int> minPathLengths = graph.GetMinPathLengths(circle,
                    cachedAllGraphPaths);
                Console.WriteLine(@"IsTauCircle = " + circle.IsTauCircle(graph, cachedAllGraphPaths));
                foreach (var pair in minPathLengths)
                {
                    Console.WriteLine("{0}=>{1}", pair.Key, pair.Value);
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
                bool result2 = _macLaneAlgorithm.IsPlanar(new Graph(graph));
                Console.WriteLine(@"{0}=={1}", result1, result2);
                Console.WriteLine();
                Assert.IsTrue(result1 == result2);
            }
        }

        [TestMethod]
        public void TestGraphFromText()
        {
            for (int i = 0; i < 20; i++)
            {
                Graph graph = Graph.Random(16, 20);
                Console.WriteLine(@"Test #" + i);
                Console.WriteLine(graph);
                Assert.IsTrue(graph.Equals(new Graph(graph.ToString())));
            }
        }

        [TestMethod]
        public void TestCudafyMacLane()
        {
            var random = new Random();
            for (int k = 0; k < 10; k++)
            {
                var matrix =
                    new BooleanMatrix(
                        Enumerable.Range(0, 5).Select(i1 => Enumerable.Range(0, 10).Select(i => random.Next()%2 == 0)));
                int[] indexes = Enumerable.Range(1, 5).Select(i => random.Next()%5).ToArray();
                CudafyMatrix.ExecuteUpdate();
                Console.WriteLine(string.Join(",",
                    CudafyMatrix.GetIndexes().Select(i => i.ToString()).ToList()));
                Console.WriteLine(string.Join(",",
                    indexes.Select(i => i.ToString()).ToList()));

                ///////////////////////////////////////////////////
                // Вычисление целевой функции обычным методом
                IEnumerable<KeyValuePair<int, int>> dictionary =
                    indexes.Select((item, value) => new KeyValuePair<int, int>(value, item));
                var matrix2 = new BooleanMatrix(
                    dictionary.Select(
                        pair1 =>
                            dictionary
                                .Where(pair2 => pair2.Value == pair1.Key && pair2.Key != pair1.Key)
                                .Select(pair => matrix[pair.Key])
                                .Aggregate(matrix[pair1.Key], BooleanVector.Xor)));

                int macLane1 = matrix2.MacLane;

                int rows = matrix.Count;
                int columns = matrix.Length;
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(
                        new ArrayOfArray<int>(matrix.Select(vector => vector.Select(b => b ? 1 : 0).ToArray()).ToArray())
                            .ToTwoDimensional());
                    CudafyMatrix.SetIndexes(indexes.ToArray());
                    CudafyMatrix.ExecuteMacLane();
                    int macLane2 = CudafyMatrix.GetMacLane();

                    CudafyMatrix.ExecuteUpdate();
                    Console.WriteLine(string.Join(",",
                        CudafyMatrix.GetIndexes().Select(i => i.ToString()).ToList()));

                    Console.WriteLine();
                    Console.WriteLine(matrix);
                    Console.WriteLine();
                    Console.WriteLine(matrix2);
                    Console.WriteLine();
                    Console.WriteLine(string.Join(Environment.NewLine,
                        Enumerable.Range(0, rows)
                            .Select(row => string.Join("", Enumerable.Range(0, columns)
                                .Select(
                                    column => CudafyMatrix.GetMatrix()[row, column].ToString())
                                .ToList())).ToList()));
                    Console.WriteLine();

                    //Console.WriteLine(string.Join(Environment.NewLine,
                    //    Enumerable.Range(0, rows)
                    //    .Select(row => string.Join("", Enumerable.Range(0, columns)
                    //        .Select(column => CudafyMatrix.GetMatrix()[row * columns + column].ToString()).ToList())).ToList()));

                    Console.WriteLine(macLane1);
                    Console.WriteLine(macLane2);
                    Assert.AreEqual(macLane1, macLane2);
                }
            }
        }

        [TestMethod]
        public void TestCudafyCanonical()
        {
            var random = new Random();
            for (int k = 0; k < 10; k++)
            {
                var matrix =
                    new BooleanMatrix(
                        Enumerable.Range(0, 5).Select(i1 => Enumerable.Range(0, 10).Select(i => random.Next()%2 == 0)));

                StackListQueue<int> list2;
                int rows = matrix.Count;
                int columns = matrix.Length;
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(
                        new ArrayOfArray<int>(matrix.Select(vector => vector.Select(b => b ? 1 : 0).ToArray()).ToArray())
                            .ToTwoDimensional());

                    CudafyMatrix.ExecuteCanonical();

                    list2 = new StackListQueue<int>(CudafyMatrix.GetIndexes()
                        .Select((first, row) => new KeyValuePair<int, int>(row, first))
                        .Where(pair => pair.Value >= 0)
                        .Select(pair => pair.Value)
                        .ToList());

                    Console.WriteLine(matrix);
                    Console.WriteLine();
                    Console.WriteLine(string.Join(Environment.NewLine,
                        Enumerable.Range(0, rows)
                            .Select(row => string.Join("", Enumerable.Range(0, columns)
                                .Select(
                                    column => CudafyMatrix.GetMatrix()[row, column].ToString())
                                .ToList())).ToList()));
                    Console.WriteLine();
                    Console.WriteLine(string.Join(Environment.NewLine,
                        CudafyMatrix.GetIndexes().Select(i => i.ToString()).ToList()));
                }

                for (int i = matrix.Count; i-- > 0;)
                {
                    BooleanVector vector = matrix.Dequeue();
                    if (vector.IsZero()) continue;
                    matrix.Enqueue(vector);
                }
                for (int i = matrix.Count; i-- > 0;)
                {
                    BooleanVector vector = matrix.Dequeue();
                    int index = vector.IndexOf(true);
                    for (int j = matrix.Count; j-- > 0;)
                    {
                        BooleanVector vector1 = matrix.Dequeue();
                        if (vector1.Count > index && vector1[index])
                        {
                            vector1 = BooleanVector.Xor(vector1, vector);
                        }
                        if (vector1.IsZero()) continue;
                        matrix.Enqueue(vector1);
                    }
                    matrix.Enqueue(vector);
                }
                Console.WriteLine("matrix:");
                Console.WriteLine(matrix);
                var list1 = new StackListQueue<int>(matrix.Select(booleanVector => booleanVector.IndexOf(true)));
                list1.Sort();
                list2.Sort();
                Console.WriteLine("list1:" + list1);
                Console.WriteLine("list2:" + list2);
                Assert.IsTrue(list1.Equals(list2));
            }
        }
    }
}