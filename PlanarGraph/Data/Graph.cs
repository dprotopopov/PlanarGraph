using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MyCudafy;
using MyCudafy.Collections;
using MyMath.GF2;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;

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

        private readonly KeyValuePairVertexVertexPathCollectionComparer _keyValuePairVertexVertexPathCollectionComparer
            =
            new KeyValuePairVertexVertexPathCollectionComparer();

        private readonly SegmentComparer _segmentComparer =
            new SegmentComparer();

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
                return this.Select(Enumerable.First)
                    .Union(this.Select(Enumerable.Last))
                    .Distinct()
                    .ToDictionary(
                        vertex => vertex,
                        vertex => new VertexSortedCollection(
                            this.Where(segment => segment.First().Equals(vertex)).
                                Select(Enumerable.Last)
                                .Union(this.Where(segment => segment.Last().Equals(vertex)).
                                    Select(Enumerable.First))));
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
        /// <param name="BooleanVector"></param>
        /// <returns></returns>
        public Graph Sugraph(BooleanVector BooleanVector)
        {
            Debug.Assert(BooleanVector.Count <= Count);
            int count = BooleanVector.Count;
            return new Graph(this.Where((item, index) => index < count && BooleanVector[index]));
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
            if (indexes[0] > 0) booleanVector.AddRange(Enumerable.Repeat(false, indexes[0]));
            booleanVector.Add(true);
            for (int i = 1; i < indexes.Count; i++)
            {
                if (indexes[i] - indexes[i - 1] > 1)
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
            var vertices = new StackListQueue<Vertex>(Vertices);

            var prev = new PathCollection[vertices.Count, vertices.Count];
            var next = new PathCollection[vertices.Count, vertices.Count];

            var read = new object();
            var write = new object();

            Parallel.ForEach(
                from i in Enumerable.Range(0, vertices.Count)
                from j in Enumerable.Range(0, vertices.Count)
                select new[] {i, j},
                pair =>
                {
                    int i = pair[0];
                    int j = pair[1];
                    lock (write) prev[i, j] = new PathCollection();
                    lock (write) next[i, j] = new PathCollection();
                });
            Parallel.ForEach(
                this,
                segment =>
                {
                    int i, j;
                    lock (read) i = vertices.IndexOf(segment.First());
                    lock (read) j = vertices.IndexOf(segment.Last());
                    lock (write) prev[i, j].Add(new Path(segment));
                    lock (write) prev[j, i].Add(new Path(segment.GetReverse()));
                });

            for (int m = 0; m < vertices.Count; m++)
            {
                Parallel.ForEach(
                    from i in Enumerable.Range(0, vertices.Count)
                    from j in Enumerable.Range(0, vertices.Count)
                    select new[] {i, j},
                    pair =>
                    {
                        PathCollection fromTo;
                        int i = pair[0];
                        int j = pair[1];
                        lock (read) fromTo = new PathCollection(prev[i, j]);
                        if (i != m && j != m)
                        {
                            PathCollection fr, to;
                            lock (read) fr = new PathCollection(prev[i, m]);
                            lock (read) to = new PathCollection(prev[m, j]);
                            var passThrow = new PathCollection(
                                from a in fr
                                from b in to
                                select new Path(a) {b.GetRange(1, b.Count - 1)});
                            passThrow.RemoveAll(path => !Path.IsSimple(new Path(path.GetRange(0, path.Count - 1))));
                            passThrow.RemoveAll(path => !Path.IsSimple(new Path(path.GetRange(1, path.Count - 1))));
                            fromTo.Add(passThrow);
                        }
                        lock (write) next[i, j] = fromTo;
                    });
                PathCollection[,] t = prev;
                prev = next;
                next = t;
            }
            var dictionaries = new Dictionary<int, PathDictionary>[vertices.Count, vertices.Count];
            Parallel.ForEach(
                from i in Enumerable.Range(0, vertices.Count)
                from j in Enumerable.Range(0, vertices.Count)
                select new[] {i, j},
                pair =>
                {
                    int i = pair[0];
                    int j = pair[1];
                    KeyValuePair<Vertex, Vertex> key;
                    PathCollection collection;
                    lock (read) key = new KeyValuePair<Vertex, Vertex>(vertices.ElementAt(i), vertices.ElementAt(j));
                    lock (read) collection = prev[i, j];
                    Dictionary<int, PathDictionary> dictionary1 = Enumerable.Range(2, vertices.Count)
                        .ToDictionary(len => len,
                            len => new PathDictionary(key, new PathCollection(collection.Where(p => p.Count == len))));
                    lock (write) dictionaries[i, j] = dictionary1;
                });
            var dictionary = new Dictionary<int, PathDictionary>();
            foreach (var pair in dictionaries)
                foreach (int i in pair.Keys)
                    if (!dictionary.ContainsKey(i)) dictionary.Add(i, pair[i]);
                    else
                        foreach (var key in pair[i].Keys)
                            if (!dictionary[i].ContainsKey(key)) dictionary[i].Add(key, pair[i][key]);
                            else dictionary[i][key].AddRange(pair[i][key]);
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
            for (Vertex v = Vertices.FirstOrDefault(vertex => this.Count(vertex.BelongsTo) == 2);
                v != null;
                v = Vertices.FirstOrDefault(vertex => this.Count(vertex.BelongsTo) == 2))
            {
                var vv = new StackListQueue<Vertex>(this.Where(v.BelongsTo).SelectMany(s => s));
                vv.RemoveAll(v.Equals);
                RemoveAll(v.BelongsTo);
                Add(new Segment(vv));
            }
        }

        /// <summary>
        ///     Удаление всех деревьев графа
        /// </summary>
        public void RemoveAllTrees()
        {
            // Удаляем по-листочку
            while (RemoveAll(segment => segment.Any(
                vertex => this.Count(vertex.BelongsTo) == 1)) != 0) ;
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
            return fromTo.Select(pair => pair.Key).Distinct(_keyValuePairVertexVertexComparer)
                .ToDictionary(
                    pair => pair,
                    pair => fromTo.Where(p2 => p2.Key.Key == pair.Key && p2.Key.Value == pair.Value).Min(p => p.Value));
        }

        public Graph GetSubgraph(IEnumerable<Vertex> vertices)
        {
            return new Graph(Children.Where(pair => vertices.Contains(pair.Key))
                .Select(pair => new KeyValuePair<Vertex, VertexSortedCollection>(pair.Key
                    , new VertexSortedCollection(pair.Value.Intersect(vertices))))
                .Where(pair => pair.Value.Any())
                .ToDictionary(pair => pair.Key, pair => pair.Value));
        }
        public Graph GetSubgraph(StackListQueue<Vertex> vertices, IEnumerable<Vertex> bridges)
        {
            return new Graph(Children.Where(pair => vertices.Contains(pair.Key))
                .Select(pair => new KeyValuePair<Vertex, VertexSortedCollection>(pair.Key
                    , new VertexSortedCollection(pair.Value.Intersect(bridges))))
                .Where(pair => pair.Value.Any())
                .ToDictionary(pair => pair.Key, pair => pair.Value));
        }

        public IEnumerable<Path> Split(Path path)
        {
            Debug.Assert(Count >= 2);
            var list = new StackListQueue<Path>();
            StackListQueue<int> indexes;
            try
            {
                int[,] matrix;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(
                        Vertices.Select(path.GetInts).Select(item => item.ToArray()).ToArray(),
                        path.GetRange(1, Count - 2).Select(path.GetInts).Select(item => item.ToArray()).ToArray()
                        );
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                    indexes = new StackListQueue<int>(CudafyMatrix.GetIndexes()
                        .Where(index => index >= 0)
                        .Select(index => index + 1));
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                indexes =
                    new StackListQueue<int>(
                        path.GetRange(1, Count - 2)
                            .Intersect(Vertices)
                            .Select(v => path.IndexOf(v)));
            }
            indexes.Sort();
            indexes.Prepend(0);
            indexes.Append(Count - 1);
            Dictionary<Vertex, VertexSortedCollection> children = Children;
            for (int prev = indexes.Dequeue(); indexes.Any(); prev = indexes.Dequeue())
            {
                if (((prev + 1) == indexes[0])
                    && children.ContainsKey(path[prev])
                    && children[path[prev]].Contains(path[indexes[0]]))
                    continue;
                list.Add(new Path(path.GetRange(prev, indexes[0] - prev + 1)));
            }
            Debug.WriteLineIf(list.Any(), path + " split by " + this + " is " +
                                          string.Join(",", list.Select(item => item.ToString())));
            return list;
        }

    }
}