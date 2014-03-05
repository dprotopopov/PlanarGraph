using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;
using PlanarGraph.Worker;
using Boolean = PlanarGraph.Types.Boolean;

namespace PlanarGraph.Algorithm
{
    /// <summary>
    ///     Гамма-алгоритм
    ///     1. (Инициализация). Выберем любой простой цикл C исходного графа G; изобразим его на плоскости в виде грани,
    ///     которую примем за уже уложенную часть G′;
    ///     сформируем сегменты Si;
    ///     если множество сегментов пусто, то перейти к п. 3.
    ///     2. (Общий шаг). Пока множество сегментов непусто:
    ///     Для каждого сегмента S найти множество Γ(S). Если существует сегмент S, для которого |Γ(S)| = 0, то граф не
    ///     планарный, конец.
    ///     Выбираем один из сегментов с минимальным числом, вмещающих его граней.
    ///     Выбираем одну из подходящих граней для выбранного сегмента.
    ///     В данном сегменте выбираем цепь между двумя контактными вершинами и укладываем ее в выбранной грани. Учтем
    ///     изменения в структуре сегментов и перейдем к п. a).
    ///     3. (Завершение). Построена плоская укладка G′ исходного графа G, конец.
    /// </summary>
    public class GammaAlgorithm : IPlanarAlgorithm
    {
        private readonly VertexComparer _vertexComparer = new VertexComparer();

        public bool IsPlanar(Graph graph)
        {
            Debug.Assert(
                graph.ChildrenOrParents.SelectMany(pair => pair.Value
                    .Select(value => graph.ChildrenOrParents.ContainsKey(value)
                                     && graph.ChildrenOrParents[value].Contains(pair.Key)))
                    .Aggregate(Boolean.And));

            if (WorkerBegin != null) WorkerBegin();
            // Шаг первый - удаляем все узлы степени 2
            // Замечание. Если на ребра планарного графа нанести произвольное число вершин степени 2, 
            // то он останется планарным; равным образом, если на ребра непланарного графа 
            // нанести вершины степени 2, то он планарным не станет.
            graph.RemoveIntermedians();

            // Шаг второй - граф нужно укладывать отдельно по компонентам связности.
            var stackListQueue = new StackListQueue<Graph> {graph.GetAllSubGraphs()};
            var keyValuePairs = new StackListQueue<KeyValuePair<Graph, Edge>>();
            var componentQueue = new StackListQueue<Graph>();
            var pathsQueue = new StackListQueue<Enum<Path>>();
            var subGraph = new Graph();
            var builded = new Graph();
            var edges = new StackListQueue<Edge>();

            while (stackListQueue.Any() || keyValuePairs.Any()
                   || componentQueue.Any() || pathsQueue.Any())
            {
                if (pathsQueue.Any())
                {
                    // На вход подаются графы, обладающие следующими свойствами:
                    // граф связный;
                    // граф имеет хотя бы один цикл;
                    // граф не имеет мостиков, т. е. ребер, после удаления которых 
                    // граф распадается на две компонеты связности.

                    Debug.WriteLine("pathsQueue " + pathsQueue);

                    StackListQueue<Path> paths = pathsQueue.Dequeue();

                    for (paths = new StackListQueue<Path> {paths.SelectMany(path => path.Split(builded)).Distinct()};
                        paths.Any();
                        paths = new StackListQueue<Path> {paths.SelectMany(path => path.Split(builded)).Distinct()})
                    {
                        // Общий шаг алгоритма следующий: 
                        // обозреваются все сегменты Si и определяются числа |Γ(Si)|. 
                        // Если хоть одно из них равно 0, то граф не планарен, конец. 
                        // Иначе, выбираем сегмент, для которого число |Γ(S)| минимально, или 
                        // один из множества, если таких сегментов несколько. 
                        // В этом сегменте найдем цепь между двумя контактными вершинами и уложим ее 
                        // в любую из граней множества Γ(S), совместив контактные вершины сегмента 
                        // с соответствующими вершинами грани. 
                        // При этом данная грань разобьется на две. 
                        // Уже уложенная часть графа G′ по количеству ребер и вершин увеличится, 
                        // а сегмент, из которого вынута цепь, исчезнет или развалится на меньшие 
                        // с новыми контактными вершинами, ведущими к вершинам G′.
                        // В результате повторения общего шага 
                        // либо будет получена плоская укладка, когда множество сегментов станет пустым, 
                        // либо будет получено, что граф G не является планарным.


                        Debug.WriteLine("Шаг алгоритма");
                        Debug.WriteLine("paths:" +
                                        string.Join(Environment.NewLine, paths.Select(path => path.ToString())));
                        Debug.WriteLine("edges:" +
                                        string.Join(Environment.NewLine, edges.Select(edge => edge.ToString())));

                        // Каждый сегмент S относительно уже построенного графа G′ представляет собой одно из двух:
                        // ребро, оба конца которого принадлежат G′, но само оно не принадлежит G′;
                        // связную компоненту графа G – G′, дополненную всеми ребрами графа G, 
                        // один из концов которых принадлежит связной компоненте, 
                        // а второй из графа G′.

                        // Для каждого сегмента S найти множество Γ(S). Если существует сегмент S, для которого |Γ(S)| = 0, то граф не
                        // планарный, конец.
                        int minCount = paths.Select(path => edges.Count(path.Belongs)).Min();
                        if (minCount == 0)
                        {
                            Debug.WriteLine("существует сегмент S, для которого |Γ(S)| = 0");
                            Debug.WriteLine("Graph:" + builded);
                            Debug.WriteLine("Paths:" +
                                            string.Join(Environment.NewLine, paths.Select(path => path.ToString())));
                            Debug.WriteLine("Edges:" +
                                            string.Join(Environment.NewLine, edges.Select(edge => edge.ToString())));
                            bool result = false;
                            if (WorkerComplite != null) WorkerComplite(result);
                            return result;
                        }
                        // Выбираем один из сегментов с минимальным числом, вмещающих его граней.
                        // Выбираем одну из подходящих граней для выбранного сегмента.
                        // В данном сегменте выбираем цепь между двумя контактными вершинами и укладываем ее в выбранной грани.
                        Debug.Assert(minCount > 0);
                        Path path1 = paths.First(path => edges.Count(path.Belongs) == minCount);
                        IEnumerable<Edge> edges1 = edges.Where(path1.Belongs);
                        Debug.WriteLine("edges1:" +
                                        string.Join(Environment.NewLine, edges1.Select(edge => edge.ToString())));
                        if (edges1.Count() > 1) Debug.WriteLine("Возможны варианты");
                        Edge edge1 = edges1.First();
                        Debug.WriteLine("path1:" + path1);
                        Debug.WriteLine("edge1:" + edge1);
                        edges.AddRange(edge1.Split(path1));
                        builded.Add(path1);
                        paths.Remove(path1);
                        edges.Remove(edge1);
                    }

                    continue;
                }

                if (componentQueue.Any())
                {
                    // Если в графе есть мосты, то их нужно разрезать, провести отдельно плоскую укладку 
                    // каждой компоненты связности, а затем соединить их мостами. 
                    // Здесь может возникнуть трудность: в процессе укладки концевые вершины моста могут 
                    // оказаться внутри плоского графа. Нарисуем одну компоненту связности, 
                    // и будем присоединять к ней другие последовательно.
                    // Каждую новую компоненту связности будем рисовать в той грани, в которой лежит 
                    // концевая вершина соответствующего моста. Так как граф связности мостами компонент 
                    // связности является деревом, мы сумеем получить плоскую укладку.

                    Graph component = componentQueue.Dequeue();
                    Debug.WriteLine("Если в графе есть мосты, то их нужно разрезать ");
                    Debug.WriteLine("component " + component);
                    Debug.WriteLine("builded " + builded);

                    pathsQueue.Add(new Enum<Path>());

                    Dictionary<Vertex, VertexEnum> componentChildren =
                        component.ChildrenOrParents.ToDictionary(keyValuePair => keyValuePair.Key,
                            keyValuePair => new VertexEnum(keyValuePair.Value));

                    List<Vertex> joins = builded.Vertices.Intersect(component.Vertices).ToList();

                    var bridgesOrSimple = new StackListQueue<Path>
                    {
                        joins
                            .SelectMany(vertex => (componentChildren[vertex]
                                .Select(vertex1 => new Path(vertex, vertex1))))
                    };

                    List<Vertex> componentJoins = joins.Intersect(component.Vertices).ToList();
                    var componentBridges = new StackListQueue<Path>
                    {
                        componentJoins
                            .SelectMany(vertex => (componentChildren[vertex]
                                .Select(vertex1 => new Path(vertex, vertex1))))
                    };

                    Debug.WriteLine("componentBridges " + componentBridges);

                    foreach (Path bridge in componentBridges)
                    {
                        componentChildren[bridge.First()].Remove(bridge.Last());
                        componentChildren[bridge.Last()].Remove(bridge.First());
                    }

                    var componentGraph = new Graph(componentChildren);

                    Debug.WriteLine("componentGraph " + componentGraph);

                    List<Vertex> componentVetices =
                        componentBridges.Select(path => path.Last()).Distinct().ToList();

                    IEnumerable<Circle> circles = componentGraph.GetAllCircles();
                    if (circles.Count() == 0)
                    {
                        Debug.WriteLine("нет циклов ");
                        // Если нарушено свойство (2), то граф — дерево и нарисовать его плоскую укладку тривиально.

                        // В это случак надо взять все пути между мостами и добавить
                        // к рассмотрению в текущем построенном графе
                        var tree = new Tree(componentGraph);
                        Dictionary<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> allPaths =
                            tree.GetAllPaths(bridgesOrSimple.Select(path => path.Last()).Distinct());
                        var list = new List<Path>();
                        foreach (Path first in bridgesOrSimple)
                            foreach (Path last in bridgesOrSimple)
                            {
                                var key = new KeyValuePair<Vertex, Vertex>(first.Last(), last.Last());
                                if (!allPaths.ContainsKey(key)) continue;
                                foreach (Path path in allPaths[key])
                                {
                                    List<Vertex> vertices = last.ToList();
                                    vertices.Reverse();
                                    list.Add(new Path(first.GetRange(0, first.Count - 1))
                                    {
                                        path.GetRange(0, path.Count - 1),
                                        vertices
                                    });
                                }
                            }

                        pathsQueue.Add(new Enum<Path>());

                        pathsQueue.Last()
                            .AddRange(
                                list.Where(path => path.Count > 3 || !path.First().Equals(path.Last())).Distinct());
                        continue;
                    }

                    //circles = circles.Where(circle => componentVetices.Select(circle.Contains)
                    //    .Aggregate(true, Boolean.And));

                    var componentEdge = new Edge(circles.First());
                    Debug.WriteLine("componentEdge " + componentEdge);

                    keyValuePairs.Add(new KeyValuePair<Graph, Edge>(componentGraph, componentEdge));

                    Debug.WriteLine("componentGraph " + componentGraph);
                    Debug.WriteLine("componentEdge " + componentEdge);

                    continue;
                }

                if (keyValuePairs.Any())
                {
                    // На вход подаются графы, обладающие следующими свойствами:
                    // граф связный;
                    // граф имеет хотя бы один цикл;
                    // граф не имеет мостиков, т. е. ребер, после удаления которых 
                    // граф распадается на две компонеты связности.

                    KeyValuePair<Graph, Edge> pair = keyValuePairs.Dequeue();
                    Graph dequeue = pair.Key;
                    edges.Add(pair.Value);
                    builded.Add(pair.Value);
                    pathsQueue.Add(new Enum<Path>());

                    Debug.WriteLine("На вход подаются графы");
                    Debug.WriteLine("builded " + builded);
                    Debug.WriteLine("edges:" +
                                    string.Join(Environment.NewLine, edges.Select(edge => edge.ToString())));

                    // Каждый сегмент S относительно уже построенного графа G′ представляет собой одно из двух:
                    // ребро, оба конца которого принадлежат G′, но само оно не принадлежит G′;
                    // связную компоненту графа G – G′, дополненную всеми ребрами графа G, 
                    // один из концов которых принадлежит связной компоненте, 
                    // а второй из графа G′.

                    var both =
                        new Graph(dequeue.Except(builded).Where(segment => segment.All(builded.Vertices.Contains)));
                    var components = new Graph(dequeue.Except(builded).Except(both));

                    List<Vertex> joins = builded.Vertices.Intersect(components.Vertices).ToList();
                    Dictionary<Vertex, VertexEnum> children =
                        components.ChildrenOrParents.ToDictionary(keyValuePair => keyValuePair.Key,
                            keyValuePair => new VertexEnum(keyValuePair.Value));

                    var bridgesOrSimple = new StackListQueue<Path>
                    {
                        joins
                            .SelectMany(vertex => (children[vertex]
                                .Select(vertex1 => new Path(vertex, vertex1))))
                    };

                    Debug.WriteLine("both " + both);
                    Debug.WriteLine("components " + components);
                    Debug.WriteLine("bridgesOrSimple " + bridgesOrSimple);

                    pathsQueue.First().AddRange(
                        both.Select(segment => new Path(segment))
                            .Where(item => !item.Belongs(builded))
                            .Distinct()
                        );

                    pathsQueue.First().AddRange(
                        bridgesOrSimple
                            .SelectMany(a => bridgesOrSimple
                                .Where(b => a.Last().Equals(b.Last()))
                                .Where(b => !a.First().Equals(b.First()))
                                .Select(b => new Path(a) {b.First()}))
                            .Distinct()
                        );


                    componentQueue = new StackListQueue<Graph> {components.GetAllSubGraphs()};

                    continue;
                }


                if (stackListQueue.Any())
                {
                    subGraph = stackListQueue.Dequeue();

                    // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
                    subGraph.RemoveAllTrees();
                    if (subGraph.Vertices.Count() < 2) continue;
                    List<Circle> subCircles = subGraph.GetAllCircles().ToList();
                    if (!subCircles.Any()) continue; // граф — дерево и нарисовать его плоскую укладку тривиально.

                    // Инициализация алгоритма производится так: выбираем любой простой цикл;
                    // и получаем две грани: Γ1 — внешнюю и Γ2 — внутреннюю

                    edges = new StackListQueue<Edge> {new Edge(subCircles.First())};
                    keyValuePairs = new StackListQueue<KeyValuePair<Graph, Edge>>
                    {
                        new KeyValuePair<Graph, Edge>(subGraph,
                            new Edge(subCircles.First()))
                    };
                    pathsQueue = new StackListQueue<Enum<Path>> {new Enum<Path>()};
                    builded = new Graph();
                }
            }
            {
                bool result = true;
                if (WorkerComplite != null) WorkerComplite(result);
                return result;
            }
        }


        public WorkerBegin WorkerBegin { get; set; }
        public WorkerComplite WorkerComplite { get; set; }
    }
}