using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.GF2;
using PlanarGraph.Parallel;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс графа без петель и кратных ребер
    ///     Граф является набором вершин и сегментов
    ///     Каждый сегмент состоит из двух вершин
    /// </summary>
    public class Graph : SegmentCollection, IElement
    {
        private readonly KeyValuePairVertexVertexComparer _keyValuePairVertexVertexComparer =
            new KeyValuePairVertexVertexComparer();

        public Graph()
        {
        }

        /// <summary>
        ///     Создание графа из строки
        ///     Строка должна содержать список сегментов, задаваемых в виде пар индексов точек
        ///     Всегда должно выполняться равенство:
        ///     graph.Equals(new Graph(graph.ToString())) == true
        /// </summary>
        /// <param name="text"></param>
        public Graph(string text)
        {
            const string segmentPattern = @"\((?<i>\d+),(?<j>\d+)\)";
            foreach (Match match in Regex.Matches(text, segmentPattern))
            {
                int i = Convert.ToInt32(match.Groups["i"].Value);
                int j = Convert.ToInt32(match.Groups["j"].Value);
                Add(new Vertex(i), new Vertex(j));
            }
        }

        public Graph(IEnumerable<Segment> segments)
            : base(segments.Distinct())
        {
        }

        public Graph(Segment segment)
            : base(segment)
        {
        }

        public Graph(IEnumerable<KeyValuePair<Vertex, VertexSortedCollection>> children)
            : base(children.SelectMany(child => child.Value.Select(vertex => new Segment(child.Key, vertex)))
                .Distinct())
        {
        }

        public Graph(Circle circle)
        {
            Add(circle);
        }

        public Graph(Edge edge)
        {
            Add(edge);
        }

        public Graph(Path path)
        {
            Add(path);
        }

        public Dictionary<Vertex, VertexSortedCollection> Children
        {
            get
            {
                return this.Select(segment => segment.First())
                    .Union(this.Select(segment => segment.Last()))
                    .Distinct()
                    .ToDictionary(
                        vertex => vertex,
                        vertex => new VertexSortedCollection(
                            this.Where(segment => segment.First().Equals(vertex)).
                                Select(segment => segment.Last())
                                .Union(this.Where(segment => segment.Last().Equals(vertex)).
                                    Select(segment => segment.First()))));
            }
        }

        public bool BelongsTo(Graph graph)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Circle circle)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Path path)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Segment segment)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Graph graph)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Circle circle)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Path path)
        {
            Dictionary<Vertex, VertexSortedCollection> children = Children;
            int count = path.Count;
            if (count == 0) return false;
            if (!children.ContainsKey(path[0])) return false;
            for (int i = 1; i < count; i++)
            {
                if (!children[path[i - 1]].Contains(path[i])) return false;
            }
            return true;
        }

        public bool FromTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        public bool FromOrTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        #region

        /// <summary>
        ///     суграф называется квазициклом, если все его вершины имеют четную валентность
        /// </summary>
        public bool IsQuasicircle
        {
            get { return Vertices.All(vertex => Children[vertex].Count%2 == 0); }
        }

        /// <summary>
        ///     Пусть G(X,U;P) - граф с множеством вершин X={x1,x2,...,xn} и ребер U={u1,u2,...,um}, где n-количество вершин графа
        ///     и m-количество ребер графа G.
        ///     Известно, что размерность подпространство квазициклов LGС совпадает с цикломатическим числом ν(G) = m - n + 1 графа
        ///     G, а порядок группы LGС равен 2**ν(G).
        /// </summary>
        public int CiclomaticNumber
        {
            get { return Count - Vertices.Count() + 1; }
        }

        /// <summary>
        ///     Пусть G(X,U;P) - граф с множеством вершин X={x1,x2,...,xn} и ребер U={u1,u2,...,um}, где n-количество вершин графа
        ///     и m-количество ребер графа G.
        ///     Пусть L - множество всех суграфов этого графа относительно операции сложения
        ///     (X,U1;P) ⊕ (X,U2;P) = (X,(U1U2)\(U1U2);P)
        ///     это множество, как известно, образует абелеву 2-группу, которую можно рассматривать как векторное пространство
        ///     над полем из двух элементов GF(2). Размерность этого пространства, называемого пространством суграфов графа G,
        ///     конечно и равно m (dim LG=m). В качестве базиса этого пространства выберем множество однореберных суграфов
        ///     (u1,u2,...,um). Тогда в этом базисе каждому элементу Y пространства LG однозначно сопоставляется последовательность
        ///     координат (a1,a2,...,am), где ai {0,1}.
        ///     При этом оказывается, что ребро ui входит в суграф Y, если ai = 1, и не входит в данный суграф - в противном
        ///     случае.
        ///     В дальнейшем для удобства будем отождествлять пространство суграфов LG и его координатное пространство.
        /// </summary>
        /// <param name="booleanVector"></param>
        /// <returns></returns>
        public Graph Sugraph(BooleanVector booleanVector)
        {
            Debug.Assert(booleanVector.Count <= Count);
            int count = booleanVector.Count;
            return new Graph(this.Where((item, index) => index < count && booleanVector[index]));
        }

        public BooleanVector GetVector(Circle circle)
        {
            Debug.Assert(circle.Any());
            int count = circle.Count;
            var collection = new SegmentCollection();
            for (int i = 0; i < count; i++)
            {
                collection.Add(new Segment(circle[i], circle[(i + 1)%count]));
            }
            Debug.Assert(collection.All(Contains));
            List<int> indexes;
            try
            {
                IEnumerable<IEnumerable<int>> list1 = collection.Select(GetInts);
                IEnumerable<IEnumerable<int>> list2 = this.Select(GetInts);
                int[,] matrix;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(
                        list1.Select(item => item.ToArray()).ToArray(),
                        list2.Select(item => item.ToArray()).ToArray()
                        );
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                    indexes = CudafyMatrix.GetIndexes().ToList();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                indexes = collection.Select(segment => IndexOf(segment)).ToList();
            }
            indexes.Sort();
            var booleanVector = new BooleanVector();
            booleanVector.AddRange(Enumerable.Repeat(false, indexes[0]));
            booleanVector.Add(true);
            for (int i = 1; i < count; i++)
            {
                booleanVector.AddRange(Enumerable.Repeat(false, indexes[i] - indexes[i - 1] - 1));
                booleanVector.Add(true);
            }
            return booleanVector;
        }

        #endregion

        #region Add

        public void Add(Vertex vertex1, Vertex vertex2)
        {
            Add(new Segment(vertex1, vertex2));
        }

        public void Add(Circle circle)
        {
            int count = circle.Count();
            AddRange(circle.Select((item, index) => new Segment(circle[index], circle[(index + 1)%count])));
        }

        public void Add(Edge edge)
        {
            int count = edge.Count();
            AddRange(edge.Select((item, index) => new Segment(edge[index], edge[(index + 1)%count])));
        }

        public void Add(Path path)
        {
            int count = path.Count();
            AddRange(
                Enumerable.Range(0, count - 1)
                    .Select(index => new Segment(path[index], path[(index + 1)%count])));
        }

        #endregion

        #region

        /// <summary>
        ///     Список узлов графа
        /// </summary>
        public VertexSortedCollection Vertices
        {
            get { return new VertexSortedCollection(Children.Keys); }
        }

        #endregion

        public override StackListQueue<int> GetInts(Segment values)
        {
            return new StackListQueue<int>(values.Select(value => value.Id));
        }

        public static Dictionary<int, PathDictionary> GetSubgraphPaths(
            IEnumerable<Vertex> vertices,
            Dictionary<int, PathDictionary> cachedAllGraphPaths)
        {
            return new Dictionary<int, PathDictionary>(GetFromToPaths(vertices, vertices, cachedAllGraphPaths)
                .ToDictionary(p => p.Key, p => new PathDictionary(p.Value
                    .Select(pair => new {pair, list1 = pair.Value.Where(path => path.All(vertices.Contains))})
                    .Where(@t => @t.list1.Any())
                    .Select(@t => new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(@t.pair.Key,
                        new PathCollection(@t.list1))))));
        }

        public static Dictionary<int, PathDictionary> GetFromToPaths(
            IEnumerable<Vertex> listFrom, IEnumerable<Vertex> listTo,
            Dictionary<int, PathDictionary> cachedAllGraphPaths)
        {
            return new Dictionary<int, PathDictionary>(cachedAllGraphPaths
                .ToDictionary(p => p.Key, p => new PathDictionary(p.Value
                    .Where(
                        pair =>
                            listFrom.Contains(pair.Key.Key) &&
                            listTo.Contains(pair.Key.Value)))));
        }

        /// <summary>
        ///     Получение всех путей в графе
        /// </summary>
        /// <returns></returns>
        public Dictionary<int, PathDictionary> GetAllGraphPaths()
        {
            Dictionary<Vertex, VertexSortedCollection> children = Children;

            var dictionary = new Dictionary<int, PathDictionary>();
            if (!Vertices.Any()) return dictionary;
            dictionary.Add(1, new PathDictionary(Vertices.Select(
                vertix =>
                    new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                        new KeyValuePair<Vertex, Vertex>(vertix, vertix),
                        new PathCollection(new Path(vertix))))));
            if (Vertices.Count() == 1) return dictionary;
            dictionary.Add(2, new PathDictionary(children.SelectMany(pair => pair.Value.Select(
                vertix => new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                    new KeyValuePair<Vertex, Vertex>(pair.Key, vertix),
                    new PathCollection(new Path(pair.Key, vertix)))))));
            if (Vertices.Count() == 2) return dictionary;
            for (int i = 3; i <= Vertices.Count() + 1; i++)
            {
                var pathDictionary = new PathDictionary();
                int i1 = (i + 1) >> 1;
                int i2 = (i + 1) - i1;
                PathDictionary dictionary1 = dictionary[i1];
                PathDictionary dictionary2 = dictionary[i2];
                foreach (Vertex vertex1 in Vertices)
                    foreach (Vertex vertex2 in Vertices)
                        foreach (Vertex vertex12 in Vertices)
                        {
                            if (vertex1.Equals(vertex12) || vertex2.Equals(vertex12)) continue;

                            var pair1 = new KeyValuePair<Vertex, Vertex>(vertex1, vertex12);
                            var pair2 = new KeyValuePair<Vertex, Vertex>(vertex12, vertex2);
                            if ((!dictionary1.ContainsKey(pair1)) || (!dictionary2.ContainsKey(pair2))) continue;
                            var pair = new KeyValuePair<Vertex, Vertex>(vertex1, vertex2);
                            PathCollection paths = pathDictionary.ContainsKey(pair)
                                ? pathDictionary[pair]
                                : new PathCollection();
                            PathCollection paths1 = dictionary1[pair1];
                            PathCollection paths2 = dictionary2[pair2];
                            Debug.WriteLine("paths1:" +
                                            string.Join(Environment.NewLine, paths1.Select(path => path.ToString())));
                            Debug.WriteLine("paths2:" +
                                            string.Join(Environment.NewLine, paths2.Select(path => path.ToString())));
                            try
                            {
                                int[,] matrix;
                                lock (CudafySequencies.Semaphore)
                                {
                                    CudafySequencies.SetSequencies(
                                        paths1.Select(
                                            path =>
                                                path.Select(vertex => vertex.Id).ToArray())
                                            .ToArray(),
                                        paths2.Select(
                                            path =>
                                                path.Select(vertex => vertex.Id).ToArray())
                                            .ToArray()
                                        );
                                    CudafySequencies.Execute("CountIntersections");
                                    matrix = CudafySequencies.GetMatrix();
                                }
#if DEBUG                      
                                for (int a = 0; a < matrix.GetLength(0); a++)
                                    for (int b = 0; b < matrix.GetLength(1); b++)
                                    {
                                        Debug.Assert(matrix[a, b] > 0);
                                    }
#endif
                                paths.AddRangeExcept(paths1.SelectMany(
                                    (values1, index1) =>
                                        paths2.Select((values2, index2) => new {index1, index2, values1, values2}))
                                    .Where(p => (matrix[p.index1, p.index2] == 1)
                                                || (i > 3 && matrix[p.index1, p.index2] == 2 &&
                                                    p.values1.First().Equals(p.values2.Last())))
                                    .Select(p =>
                                        new Path(p.values1.GetRange(0, i1 - 1)) {p.values2}));
                                Debug.WriteLine("paths:" +
                                                string.Join(Environment.NewLine, paths.Select(path => path.ToString())));
                            }
                            catch (Exception ex)
                            {
                                Debug.WriteLine(ex.ToString());
                                paths.AddRangeExcept(from path1 in paths1
                                    from path2 in paths2
                                    where !path1.GetRange(0, i1 - 1).Intersect(path2.GetRange(0, i2 - 1)).Any()
                                          && !path1.GetRange(1, i1 - 1).Contains(path2.Last())
                                    select new Path(path1.GetRange(0, i1 - 1)) {path2});
                            }
                            paths.ReplaceAll(paths.Distinct());
                            if (!paths.Any()) continue;
                            if (pathDictionary.ContainsKey(pair)) pathDictionary[pair].AddRangeExcept(paths);
                            else pathDictionary.Add(pair, paths);
                        }
                if (!pathDictionary.Any()) break;
                dictionary.Add(i, pathDictionary);
            }
            return dictionary;
        }

        public override bool Equals(object obj)
        {
            var graph = obj as Graph;
            return graph != null && Vertices.Equals(graph.Vertices) && base.Equals(graph);
        }

        public override int GetHashCode()
        {
            return Vertices.GetHashCode() ^ base.GetHashCode();
        }

        public static Graph Random(int n, int m)
        {
            var graph = new Graph();
            var random = new Random(DateTime.Now.GetHashCode());
            while (graph.Count < Math.Min(m, n*(n - 1)/2))
            {
                int i = random.Next()%n;
                int j = random.Next()%n;
                while (j == i) j = random.Next()%n;
                graph.Add(new Vertex(i), new Vertex(j));
            }
            Debug.Assert(
                graph.Vertices.All(key => graph.Children[key]
                    .All(value => graph.Children.ContainsKey(value)))
                );
            Debug.Assert(
                graph.Vertices.All(key => graph.Children[key]
                    .All(value => graph.Children[value].Contains(key)))
                );
            return graph;
        }

        public void RemoveIntermedians()
        {
            for (List<Segment> segments = this.SelectMany(segment => segment)
                .Distinct()
                .Where(vertix => this.Count(segment2 => segment2.Contains(vertix)) == 1)
                .SelectMany(vertix => this.Where(segment2 => segment2.Contains(vertix)))
                .ToList();
                segments.Any();
                segments = this.SelectMany(segment => segment)
                    .Distinct()
                    .Where(vertix => this.Count(segment2 => segment2.Contains(vertix)) == 1)
                    .SelectMany(vertix => this.Where(segment2 => segment2.Contains(vertix)))
                    .ToList()
                )
            {
                foreach (Segment segment in segments)
                    Remove(segment);
            }
        }

        /// <summary>
        ///     Удаление всех деревьев графа
        /// </summary>
        public void RemoveAllTrees()
        {
            for (List<Segment> segments = this.Where(
                segment2 => segment2.Any(
                    vertex => this.Count(segment1 => segment1.Contains(vertex)) == 1)).ToList();
                segments.Any();
                segments = this.Where(
                    segment2 => segment2.Any(
                        vertex => this.Count(segment1 => segment1.Contains(vertex)) == 1)).ToList())
                foreach (Segment segment in segments)
                    Remove(segment);
        }

        public override string ToString()
        {
            return String.Join(":",
                new[]
                {
                    Vertices.ToString(),
                    base.ToString()
                });
        }

        /// <summary>
        ///     Разделение графа на связанные подграфы
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Graph> GetAllSubGraphs()
        {
            var list = new StackListQueue<Graph>();
            List<Vertex> vertexes = Vertices.ToList();
            while (vertexes.Any())
            {
                var stackListQueue = new StackListQueue<Vertex> {vertexes.First()};
                var list1 = new VertexUnsortedCollection();
                while (stackListQueue.Any())
                {
                    Vertex pop = stackListQueue.Pop();
                    vertexes.Remove(pop);
                    list1.Add(pop);
                    stackListQueue.AddRange(Children[pop].Intersect(vertexes).Except(stackListQueue));
                }
                Dictionary<Vertex, VertexSortedCollection> children = list1.ToDictionary(vertex => vertex,
                    vertex => Children[vertex]);
                list.Add(new Graph(children));
            }
            return list;
        }

        /// <summary>
        ///     Поиск всех циклов в графе
        ///     Длина цикла должна быть больше двух
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Circle> GetAllGraphCircles(
            Dictionary<int, PathDictionary> cachedAllGraphPaths)
        {
            var graph1 = new Graph(this);
            graph1.RemoveAllTrees();
            IEnumerable<Circle> list = (from p in GetSubgraphPaths(graph1.Vertices, cachedAllGraphPaths)
                where p.Key > 3
                from pair in p.Value
                where pair.Key.Key.Equals(pair.Key.Value)
                from path in pair.Value
                select new Circle(path.GetRange(0, path.Count - 1)));
            return list.Distinct();
        }

        /// <summary>
        ///     Поиск длин минимальных путей в графе, соединяющих любые две точки
        /// </summary>
        /// <returns></returns>
        public Dictionary<KeyValuePair<Vertex, Vertex>, int> GetMinPathLengths(IEnumerable<Vertex> vertices,
            Dictionary<int, PathDictionary> cachedAllGraphPaths)
        {
            Debug.Assert(Vertices.Any());
            IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, int>> fromTo = GetFromToPaths(vertices, vertices,
                cachedAllGraphPaths)
                .SelectMany(
                    p => p.Value.Select(pair =>
                        new KeyValuePair<KeyValuePair<Vertex, Vertex>, int>(pair.Key, p.Key)));

            return fromTo.Where(
                p1 =>
                    p1.Value ==
                    fromTo.Where(p2 => p2.Key.Key == p1.Key.Key && p2.Key.Value == p1.Key.Value).Min(p => p.Value))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
        }

        public Graph GetSubgraph(IEnumerable<Vertex> vertices)
        {
            return new Graph(Children.Where(pair => vertices.Contains(pair.Key))
                .Select(pair => new KeyValuePair<Vertex, VertexSortedCollection>(pair.Key
                    , new VertexSortedCollection(pair.Value.Intersect(vertices))))
                .Where(pair => pair.Value.Any())
                .ToDictionary(pair => pair.Key, pair => pair.Value));
        }
    }
}