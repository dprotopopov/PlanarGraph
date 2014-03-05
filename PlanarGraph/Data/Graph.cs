using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.GF2;
using Boolean = PlanarGraph.Types.Boolean;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс графа без петель и кратных ребер
    ///     Граф является набором вершин и сегментов
    ///     Каждый сегмент состоит из двух вершин
    /// </summary>
    public class Graph : SegmentEnum
    {
        protected readonly VertexComparer _vertexComparer = new VertexComparer();

        public Graph()
        {
        }

        public Graph(IEnumerable<Segment> segments)
        {
            AddRange(segments.Distinct().ToList());
        }

        public Graph(IEnumerable<KeyValuePair<Vertex, VertexEnum>> children)
        {
            AddRange(children.SelectMany(child => child.Value.Select(vertex => new Segment(child.Key, vertex)))
                .Distinct());
        }

        public Graph(Circle circle)
        {
            Add(circle);
        }

        public Graph(Edge edge)
        {
            Add(edge);
        }

        public Graph(Graph graph)
        {
            AddRange(graph);
        }

        public Dictionary<Vertex, VertexEnum> ChildrenOrParents
        {
            get
            {
                return this.Select(segment => segment.First())
                    .Union(this.Select(segment => segment.Last()))
                    .Distinct().ToDictionary(
                        vertex => vertex,
                        vertex => new VertexEnum(
                            this.Where(segment => segment.First().Equals(vertex)).
                                Select(segment => segment.Last())
                                .Union(this.Where(segment => segment.Last().Equals(vertex)).
                                    Select(segment => segment.First()))));
            }
        }


        /// <summary>
        ///     Проверка, что граф связный
        /// </summary>
        public bool IsConnected
        {
            get { return GetAllSubGraphs().Count() == 1; }
        }

        /// <summary>
        ///     Проверка, что граф имеет хотя бы один цикл
        /// </summary>
        public bool HasCircle
        {
            get { return GetAllCircles().Any(); }
        }

        /// <summary>
        ///     граф не имеет мостов, т. е. ребер, после удаления которых, граф распадается на две компоненты связности
        /// </summary>
        public bool HasNoBridges
        {
            get
            {
                var graph = new Graph(this);
                graph.RemoveAllTrees();

                var graph1 = new Graph();
                foreach (Circle circle in GetAllCircles())
                    graph1.Add(circle);

                return !graph.Except(graph1).Any();
            }
        }

        private IEnumerable<Circle> AllCircles { get; set; }
        private IEnumerable<Path> AllBranches { get; set; }
        private IEnumerable<Graph> AllSubGraphs { get; set; }

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
                graph.ChildrenOrParents.SelectMany(pair => pair.Value
                    .Select(value => graph.ChildrenOrParents.ContainsKey(value)
                                     && graph.ChildrenOrParents[value].Contains(pair.Key)))
                    .Aggregate(true, Boolean.And));
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

            AllBranches = null;
            AllSubGraphs = null;
            AllCircles = null;
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

            AllBranches = null;
            AllSubGraphs = null;
            AllCircles = null;
        }

        /// <summary>
        ///     Нахождение всех мостов графа
        ///     Мосты представляют собой деревья
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Graph> GetAllBridges()
        {
            var list = new List<Graph>();
            foreach (Graph graph in GetAllSubGraphs())
            {
                graph.RemoveAllTrees();
                list.AddRange(
                    new Graph(
                        graph.Except(
                            graph.GetAllCircles()
                                .Select(circle => new Graph(circle))
                                .SelectMany(circlegraph => circlegraph)
                                .Distinct())).GetAllSubGraphs());
            }
            return list;
        }

        public override string ToString()
        {
            return string.Join(Environment.NewLine,
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
            if (AllSubGraphs != null) return AllSubGraphs;
            var list = new List<Graph>();
            List<Vertex> vertexes = Vertices.ToList();
            while (vertexes.Any())
            {
                var stackListQueue = new StackListQueue<Vertex> {vertexes.First()};
                var list1 = new VertexCollection();
                while (stackListQueue.Any())
                {
                    Vertex pop = stackListQueue.Pop();
                    vertexes.Remove(pop);
                    list1.Add(pop);
                    stackListQueue.AddRange(ChildrenOrParents[pop].Intersect(vertexes).Except(stackListQueue));
                }
                Dictionary<Vertex, VertexEnum> children = list1.ToDictionary(vertex => vertex,
                    vertex => ChildrenOrParents[vertex]);
                list.Add(new Graph(children));
            }
            AllSubGraphs = list;
            return AllSubGraphs;
        }

        public Graph GetSubgraph(IEnumerable<Vertex> vertices)
        {
            return new Graph(this.Where(segment => segment.All(vertices.Contains)));
        }

        /// <summary>
        ///     Поиск всех циклов в графе
        ///     Длина цикла должна быть больше двух
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Circle> GetAllCircles()
        {
            if (AllCircles != null) return AllCircles;
            var list = new List<Circle>();
            var stackListQueue = new StackListQueue<Path>();

            var graph1 = new Graph(this);
            graph1.RemoveAllTrees();
            foreach (
                var children in
                    graph1.GetAllSubGraphs()
                        .Select(graph => graph.Vertices)
                        .Select(collection => collection.ToDictionary(vertex => vertex,
                            vertex => new VertexEnum(ChildrenOrParents[vertex].Intersect(collection))))
                )
            {
                // Граф, представленный children является одной компонентой связанности
                stackListQueue.Add(new Path(children.Keys.First()));
                while (stackListQueue.Any())
                {
                    Path pop = stackListQueue.Dequeue();
                    int count = pop.Count;
                    Debug.Assert(children.ContainsKey(pop.Last()));
                    if (children[pop.First()].Any())
                    {
                        List<Vertex> list2 = children[pop.First()].Intersect(pop).ToList();
                        List<int> indexes2 =
                            list2.Select(vertex =>
                                pop.IndexOf(vertex)).ToList();
                        list.AddRange(
                            list2.Select(
                                (vertex, index) =>
                                    new Circle(pop.GetRange(0, indexes2[index] + 1)))
                                .Where(circle => circle.Count > 2)
                                .Except(list));

                        stackListQueue.AddRange(
                            children[pop.First()].Except(pop).Select(
                                vertex =>
                                    new Path(vertex) {pop})
                                .Except(stackListQueue)
                            );
                    }
                    if (!children[pop.Last()].Any())
                    {
                        List<Vertex> list1 = children[pop.Last()].Intersect(pop).ToList();
                        List<int> indexes1 =
                            list1.Select(vertex =>
                                pop.IndexOf(vertex)).ToList();
                        list.AddRange(
                            list1.Select(
                                (vertex, index) =>
                                    new Circle(pop.GetRange(indexes1[index], count - indexes1[index])))
                                .Where(circle => circle.Count > 2)
                                .Except(list));
                        stackListQueue.AddRange(
                            children[pop.Last()].Except(pop).Select(
                                vertex =>
                                    new Path(pop) {vertex})
                                .Except(stackListQueue)
                            );
                    }
                }
            }
            AllCircles = list.Distinct().ToList();
            return AllCircles;
        }

        /// <summary>
        ///     Поиск минимальных путей в графе, соединяющих любые две точки
        ///     Условие поиска - граф должен быть связанным, точки должны принадлежать графу
        /// </summary>
        /// <returns></returns>
        public Dictionary<KeyValuePair<Vertex, Vertex>, List<Path>> GetMinPaths(IEnumerable<Vertex> vertices)
        {
            Debug.Assert(Vertices.Any());
            Debug.Assert(vertices.Select(Vertices.Contains).Aggregate(true, Boolean.And));
            Debug.Assert(vertices.Distinct().Count() == vertices.Count());
            var getMinPaths = new Dictionary<KeyValuePair<Vertex, Vertex>, List<Path>>();
            // Случай 1.  Узлы лежат на одной ветви
            // Случай 2.  Узлы либо лежат на разных ветвях, либо на циклах
            // Узлы могут лежать на ветвях, не входящих в циклы
            // Мы рассматриваем только узлы лежащие на циклах
            var stackListQueue = new StackListQueue<Path>();

            var graph1 = new Graph(this);
            graph1.RemoveAllTrees();

            foreach (
                var children in
                    GetAllCircles()
                        .Select(circle => circle.ToList())
                        .Where(collection => vertices.Select(collection.Contains).Count() >= 2)
                        .Select(collection => collection.ToDictionary(vertex => vertex,
                            vertex => new VertexEnum(ChildrenOrParents[vertex].Intersect(collection))))
                )
            {
                // Граф, представленный children является одной компонентой связанности

                List<Vertex> second = children.Keys.Intersect(vertices).ToList();
                if (!second.Any()) continue;
                Dictionary<KeyValuePair<Vertex, Vertex>, List<Path>> dictionary = children.Keys.ToDictionary
                    (
                        key => new KeyValuePair<Vertex, Vertex>(key, key),
                        key => new List<Path> {new Path(key)}
                    );
                foreach (
                    var keys in
                        from pair in children
                        from vertex in pair.Value
                        select new List<Vertex> {pair.Key, vertex})
                {
                    keys.Sort(_vertexComparer);
                    var key = new KeyValuePair<Vertex, Vertex>(keys[0], keys[1]);
                    if (!dictionary.ContainsKey(key))
                        dictionary.Add(key, new List<Path> {new Path {keys}});
                }

                stackListQueue.AddRange(second.Select(vertex => new Path(vertex)));

                while (stackListQueue.Any())
                {
                    Path pop = stackListQueue.Dequeue();
                    Debug.Assert(children.ContainsKey(pop.Last()));
                    Debug.Assert(children.ContainsKey(pop.First()));

                    {
                        // Проверяем является ли данный путь, соединяющий две точки
                        // минимальным путём. И если ранее был найден путь более короткий,
                        // то пропускаем цикл
                        var keys = new List<Vertex> {pop.First(), pop.Last()};
                        keys.Sort(_vertexComparer);
                        var key = new KeyValuePair<Vertex, Vertex>(keys[0], keys[1]);
                        if (!dictionary.ContainsKey(key)) dictionary.Add(key, new List<Path> {pop});
                        else if (dictionary[key].First().Count > pop.Count)
                            dictionary[key] = new List<Path> {pop};
                        else if (dictionary[key].First().Count == pop.Count && !dictionary[key].Contains(pop))
                            dictionary[key].Add(pop);
                        else if (dictionary[key].First().Count < pop.Count) continue;
                    }

                    stackListQueue.AddRange(
                        children[pop.First()].Except(pop).Select(
                            vertex =>
                                new Path(vertex) {pop})
                            .Except(stackListQueue)
                        );
                    stackListQueue.AddRange(
                        children[pop.Last()].Except(pop).Select(
                            vertex =>
                                new Path(pop) {vertex})
                            .Except(stackListQueue)
                        );
                }

                foreach (var pair in (from first in second
                    from last in second
                    where _vertexComparer.Compare(first, last) < 0
                    select new KeyValuePair<Vertex, Vertex>(first, last)))
                    if (dictionary.ContainsKey(pair))
                        foreach (Path path in dictionary[pair])
                        {
                            var keys = new List<Vertex> {path.First(), path.Last()};
                            keys.Sort(_vertexComparer);
                            var key = new KeyValuePair<Vertex, Vertex>(keys[0], keys[1]);
                            if (!getMinPaths.ContainsKey(key)) getMinPaths.Add(key, new List<Path> {path});
                            else if (getMinPaths[key].First().Count > path.Count)
                                getMinPaths[key] = new List<Path> {path};
                            else if (getMinPaths[key].First().Count == path.Count &&
                                     !getMinPaths[key].Contains(path))
                                getMinPaths[key].Add(path);
                        }
            }
            return getMinPaths;
        }

        #region

        /// <summary>
        ///     суграф называется квазициклом, если все его вершины имеют четную валентность
        /// </summary>
        public bool IsQuasicircle
        {
            get
            {
                return Vertices.Select(vertex => ChildrenOrParents[vertex].Count%2 == 0).Aggregate(true, Boolean.And);
            }
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
            var collection = new SegmentEnum();
            for (int i = 0; i < count; i++)
            {
                collection.Add(new Segment(circle[i], circle[(i + 1)%count]));
            }
            Debug.Assert(collection.Select(Contains).Aggregate(true, Boolean.And));
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
            if (!vertex1.Equals(vertex2)) base.Add(new Segment(vertex1, vertex2));
        }

        public new void Add(IEnumerable<Segment> segments)
        {
            List<Segment> collection = segments.Distinct().Except(this).ToList();
            AddRange(collection);
        }

        public void Add(IEnumerable<Path> paths)
        {
            foreach (Path path in paths)
                Add(path);
        }

        public void Add(Circle circle)
        {
            int count = circle.Count();
            for (int i = 0; i < count; i++)
            {
                Add(circle[i], circle[(i + 1)%count]);
            }
        }

        private void Add(Edge edge)
        {
            int count = edge.Count();
            for (int i = 0; i < count; i++)
            {
                Add(edge[i], edge[(i + 1)%count]);
            }
        }

        public void Add(Path path)
        {
            int count = path.Count();
            for (int i = 0; i < count - 1; i++)
            {
                Add(path[i], path[i + 1]);
            }
        }

        #endregion

        #region

        /// <summary>
        ///     Список узлов графа
        /// </summary>
        public VertexEnum Vertices
        {
            get { return new VertexEnum(ChildrenOrParents.Keys); }
        }

        #endregion
    }
}