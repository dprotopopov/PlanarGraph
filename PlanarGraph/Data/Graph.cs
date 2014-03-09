using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.GF2;

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


        public static PathDictionary GetSubgraphPaths(
            IEnumerable<Vertex> vertices,
            IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>> cachedAllGraphPaths)
        {
            return new PathDictionary(GetFromToPaths(vertices, vertices, cachedAllGraphPaths)
                .Select(pair => new {pair, list1 = pair.Value.Where(path => path.All(vertices.Contains))})
                .Where(@t => @t.list1.Any())
                .Select(@t => new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(@t.pair.Key,
                    new PathCollection(@t.list1))));
        }

        public static PathDictionary GetFromToPaths(
            IEnumerable<Vertex> listFrom, IEnumerable<Vertex> listTo,
            IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>> cachedAllGraphPaths)
        {
            return new PathDictionary(cachedAllGraphPaths
                .Where(
                    pair =>
                        listFrom.Contains(pair.Key.Key) &&
                        listTo.Contains(pair.Key.Value)));
        }

        /// <summary>
        ///     Получение всех путей в графе
        /// </summary>
        /// <returns></returns>
        public PathDictionary GetAllGraphPaths()
        {
            var stackListQueue =
                new StackListQueue<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>>
                    (
                    Vertices.Select(
                        vertix =>
                            new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                                new KeyValuePair<Vertex, Vertex>(vertix, vertix),
                                new PathCollection(new Path(vertix))))
                    );

            var collection =
                new StackListQueue<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>>();

            Dictionary<Vertex, VertexSortedCollection> children = Children;

            while (stackListQueue.Any())
            {
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> pair = stackListQueue.Dequeue();
                    collection.Add(pair);
                    if (children.ContainsKey(pair.Key.Key))
                        foreach (Vertex first in children[pair.Key.Key])
                            if (first.Equals(pair.Key.Value))
                                collection.AddExcept(new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                                    new KeyValuePair<Vertex, Vertex>(first, pair.Key.Value),
                                    new PathCollection
                                        (
                                        pair.Value.Where(path => path.Last().Equals(first))
                                            .Select(path => new Path(first) {path})
                                            .Distinct()
                                        )));
                            else
                                stackListQueue.AddExcept(new KeyValuePair
                                    <KeyValuePair<Vertex, Vertex>, PathCollection>(
                                    new KeyValuePair<Vertex, Vertex>(first, pair.Key.Value),
                                    new PathCollection
                                        (
                                        pair.Value.Where(path => !path.Contains(first))
                                            .Select(path => new Path(first) {path})
                                            .Distinct()
                                        )));
                    if (children.ContainsKey(pair.Key.Value))
                        foreach (Vertex last in children[pair.Key.Value])
                            if (last.Equals(pair.Key.Key))
                                collection.AddExcept(new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                                    new KeyValuePair<Vertex, Vertex>(pair.Key.Key, last),
                                    new PathCollection
                                        (
                                        pair.Value.Where(path => path.First().Equals(last))
                                            .Select(path => new Path(path) {last})
                                            .Distinct()
                                        )));
                            else
                                stackListQueue.AddExcept(new KeyValuePair
                                    <KeyValuePair<Vertex, Vertex>, PathCollection>(
                                    new KeyValuePair<Vertex, Vertex>(pair.Key.Key, last),
                                    new PathCollection
                                        (
                                        pair.Value.Where(path => !path.Contains(last))
                                            .Select(path => new Path(path) {last})
                                            .Distinct()
                                        )));
                }
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> pair = stackListQueue.Dequeue();
                    stackListQueue.Enqueue(pair);
                    if (pair.Key.Key.Equals(pair.Key.Value)) continue;
                    stackListQueue.AddExcept(new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                        new KeyValuePair<Vertex, Vertex>(pair.Key.Value, pair.Key.Key),
                        new PathCollection {pair.Value.Select(path => new Path(path.GetReverse()))}));
                }

                collection = new StackListQueue<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>>
                    (
                    collection.Where(pair => pair.Value.Any())
                    );
                stackListQueue = new StackListQueue<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>>
                    (
                    stackListQueue.Where(pair => pair.Value.Any())
                    );
                collection.Sort(_keyValuePairVertexVertexComparer);
                stackListQueue.Sort(_keyValuePairVertexVertexComparer);

                KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> pair1 = collection.Dequeue();
                pair1 = new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                    pair1.Key, new PathCollection(pair1.Value.Distinct()));
                for (int i = collection.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> pair2 =
                        collection.Dequeue();
                    if (_keyValuePairVertexVertexComparer.Compare(pair1, pair2) == 0)
                    {
                        pair1 = new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                            pair1.Key, new PathCollection(pair1.Value.Union(pair2.Value).Distinct()));
                    }
                    else
                    {
                        collection.Enqueue(pair1);
                        pair1 = pair2;
                    }
                }
                collection.Enqueue(pair1);
                if (!stackListQueue.Any()) break;

                pair1 = stackListQueue.Dequeue();
                pair1 = new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                    pair1.Key, new PathCollection(pair1.Value.Distinct()));
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> pair2 =
                        stackListQueue.Dequeue();
                    if (_keyValuePairVertexVertexComparer.Compare(pair1, pair2) == 0)
                    {
                        pair1 = new KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>(
                            pair1.Key, new PathCollection(pair1.Value.Union(pair2.Value).Distinct()));
                    }
                    else
                    {
                        stackListQueue.Enqueue(pair1);
                        pair1 = pair2;
                    }
                }
                stackListQueue.Enqueue(pair1);
            }
            return new PathDictionary(collection.Where(pair=>pair.Value.Any()));
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
            while (graph.Count < m)
            {
                int i = random.Next(n);
                int j = random.Next(n);
                while (j == i) j = random.Next(n);
                graph.Add(new Vertex(i), new Vertex(j));
            }
            Debug.Assert(
                graph.Children.All(pair => pair.Value
                    .All(value => graph.Children.ContainsKey(value)
                                  && graph.Children[value].Contains(pair.Key)))
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
            var list = new List<Graph>();
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
            IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>> cachedAllGraphPaths)
        {
            var graph1 = new Graph(this);
            graph1.RemoveAllTrees();
            IEnumerable<Circle> list = (from pair in cachedAllGraphPaths
                where pair.Key.Key.Equals(pair.Key.Value)
                from path in pair.Value
                where path.Count > 3
                select new Circle(path.GetRange(0, path.Count - 1)));
            return list.Distinct();
        }

        /// <summary>
        ///     Поиск минимальных путей в графе, соединяющих любые две точки
        /// </summary>
        /// <returns></returns>
        public Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> GetMinPaths(IEnumerable<Vertex> vertices,
            IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>> cachedAllGraphPaths)
        {
            Debug.Assert(Vertices.Any());
            return
                cachedAllGraphPaths.Where(pair => vertices.Contains(pair.Key.Key) && vertices.Contains(pair.Key.Value))
                    .ToDictionary(pair => pair.Key, pair => pair.Value);
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
            List<int> indexes = collection.Select(segment => IndexOf(segment)).ToList();
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
                    .Select((item, index) => new Segment(path[index], path[(index + 1)%count])));
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