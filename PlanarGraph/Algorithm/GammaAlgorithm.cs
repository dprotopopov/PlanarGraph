using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MyCudafy;
using MyCudafy.Collections;
using MyLibrary.Worker;
using PlanarGraph.Collections;
using PlanarGraph.Data;

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
    ///     Гамма алгоритм:
    ///     Удаляются все листья
    ///     Удаляются промежуточные точки
    ///     Исходный граф разбивается на связанные подграфы к которым собственно и применяется алгоритм
    ///     Создаём очередь подграфов - в начале в ней только один подграф
    ///     Выбираем из очереди один подграф
    ///     Если мы ещё не начинали рисовать то берем любой цикл и дважду помещаем его в множество граней
    ///     Если мы продолжаем рисовать внутри уже нарисованного графа, то берём один любой цикл и помещаем его один раз в
    ///     множество граней
    ///     Если цикла нет - то даннай граф -дерево то проверяем, что все пути укаладываются в уже построенный граф
    ///     Добавлем последнюю добавленную грань в множество граней к уже построенному графу
    ///     Разбиваем текущий граф на 2 графа - граф проходящий только через точки построенного графа и граф проходящий через
    ///     точки не принадлежащие построенному графу
    ///     граф проходящий через точки не принадлежащие построенному графу разбиваем на связанные подграфы и отправляем в
    ///     очередь для дальнейшего рассмотрения
    ///     Берём все пути из множества точек построенного графа в множество точек построенного графа - эти пути прохлдят как
    ///     через сам построенный граф, так и через точки не принадлежащие построенному графу.
    ///     Рассматривать пути ведущие из множества точек построенного графа в множество точек не принадлежащих построенному
    ///     графу не нужно, поскольку
    ///     либо пара таких путей проходит через разные мосты к связанной компоненте, могут быть там соединены через связанную
    ///     компоненту, а значит полученный путь уже в множестве рассматриваемых путей
    ///     либо этот путь проходит только через единичный мост содиняющий построенный граф с некой связанной компонентой, а
    ///     единичные мосты нужно удалить из дальнейшей проверки
    ///     И только теперь применяем к этим путям собственно то, что описано
    ///     Последовательно наращиваем построенный граф путями из данного множества, каждый раз пересматривая это множество
    ///     разбивая пути на подпути если путь пересекается с уже построенным графом и удаляя дубликаты от этого разбиения
    ///     То есть каждый путь должен двумя концами лежать на одной грани - если найден путь не удовлетворяющий такому
    ///     условию, то весь граф не планарен
    ///     Берём любой путь с минимальным количеством граней к которым он примыкает и любая грань к которой примыкает путь.
    ///     Разбиваем грань на две грани и добавляем их в множество граней, а исходную грань исключаем из множества граней
    ///     Добавляем этот путь к построенному графу
    ///     и повторяем пока множество оставшихся путей не пусто - то есть опять разбиваем их построенным графом и т.д.
    ///     Переходим к следубщему подграфу из очереди и повторяем
    /// </summary>
    public class GammaAlgorithm : IPlanarAlgorithm
    {
        public bool IsPlanar(Graph graphArgument)
        {
            if (WorkerLog != null) WorkerLog("Начало гамма-алгоритма");
            var graph = new Graph(graphArgument);
            Debug.Assert(
                graph.Children.All(pair => pair.Value
                    .All(value => graph.Children.ContainsKey(value)
                                  && graph.Children[value].Contains(pair.Key)))
                );

            if (WorkerBegin != null) WorkerBegin();
            // Шаг первый - удаляем все листья и узлы степени 2
            if (WorkerLog != null) WorkerLog("Удаляем все листья и узлы степени 2");

            // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
            graph.RemoveAllTrees();

            // Замечание. Если на ребра планарного графа нанести произвольное число вершин степени 2, 
            // то он останется планарным; равным образом, если на ребра непланарного графа 
            // нанести вершины степени 2, то он планарным не станет.
            graph.RemoveIntermedians();

            // Шаг второй - граф нужно укладывать отдельно по компонентам связности.
            if (WorkerLog != null) WorkerLog("Находим ВСЕ пути в графе длины не более размера графа + 1");
            Dictionary<int, PathDictionary> cachedAllGraphPaths = graph.GetAllGraphPaths();
            var queue = new StackListQueue<Context>
            {
                graph.GetAllSubGraphs().Select(subgraph =>
                    new Context
                    {
                        SubGraphQueue =
                            new StackListQueue<Graph>
                            {
                                subgraph
                            },
                        CachedSubGraphPathsQueue =
                            new StackListQueue<Dictionary<int, PathDictionary>>
                            {
                                Graph.GetSubgraphPaths(subgraph.Vertices, cachedAllGraphPaths)
                            },
                    })
            };

            Debug.WriteLine("start ");

            foreach (Context context in queue)
            {
                while (context.SubGraphQueue.Any())
                {
                    Graph subGraph = context.SubGraphQueue.Dequeue();
                    if (WorkerLog != null) WorkerLog("Проверка связанной компоненты " + subGraph);
                    Dictionary<int, PathDictionary> cachedSubGraphPaths =
                        context.CachedSubGraphPathsQueue.Dequeue();

                    // На вход подаются графы, обладающие следующими свойствами:
                    // граф связный;
                    // граф имеет хотя бы один цикл;
                    // граф не имеет мостиков, т. е. ребер, после удаления которых 
                    // граф распадается на две компонеты связности.

                    if (WorkerLog != null)
                        WorkerLog(
                            "Находим мосты после удаления которых граф распадается на несколько компонет связности");

                    var vertices = new StackListQueue<Vertex>(subGraph.Vertices);
                    var bridges = new StackListQueue<Vertex>();
                    for (int i = 0; i < vertices.Count; i++)
                    {
                        Vertex dequeue = vertices.Dequeue();
                        IEnumerable<Graph> subsubgraphs = subGraph.GetSubgraph(vertices).GetAllSubGraphs();
                        if (subsubgraphs.Count() > 1) bridges.Add(dequeue);
                        vertices.Enqueue(dequeue);
                    }

                    Debug.Assert(bridges.Count != vertices.Count);

                    if (bridges.Any())
                    {
                        // Если в графе есть мосты, то их нужно разрезать, провести отдельно плоскую укладку 
                        // каждой компоненты связности, а затем соединить их мостами. 
                        // Здесь может возникнуть трудность: в процессе укладки концевые вершины моста могут 
                        // оказаться внутри плоского графа. Нарисуем одну компоненту связности, 
                        // и будем присоединять к ней другие последовательно.
                        // Каждую новую компоненту связности будем рисовать в той грани, в которой лежит 
                        // концевая вершина соответствующего моста. Так как граф связности мостами компонент 
                        // связности является деревом, мы сумеем получить плоскую укладку.
                        if (WorkerLog != null)
                            WorkerLog(
                                "В графе есть мосты, их нужно разрезать, провести отдельно плоскую укладку, а затем соединить их мостами.");
                        if (WorkerLog != null)WorkerLog("Мосты: "+string.Join(",",bridges));
                        
                        IEnumerable<Vertex> exceptBridges = vertices.Except(bridges);
                        IEnumerable<Graph> subsubgraphs = subGraph.GetSubgraph(exceptBridges).GetAllSubGraphs();
                        context.SubGraphQueue.Enqueue(subsubgraphs.Select(subgraph => subGraph.GetSubgraph(subgraph.Vertices.Union(bridges))));
                        context.CachedSubGraphPathsQueue.Enqueue(
                            subsubgraphs.Select(
                                subgraph => Graph.GetSubgraphPaths(subgraph.Vertices.Union(bridges), cachedSubGraphPaths)));
                        
                        continue;
                    }

                    if (WorkerLog != null) WorkerLog("Находим ЛЮБОЙ МАКСИМАЛЬНОЙ ДЛИНЫ простой цикл в графе");
                    Circle circle = null;
                    for (int i = cachedSubGraphPaths.Keys.Max(); i > 3; i--)
                    {
                        foreach (var pair in cachedSubGraphPaths.Where(pair => pair.Key == i))
                        {
                            foreach (
                                var key in
                                    subGraph.Vertices.Select(vertex => new KeyValuePair<Vertex, Vertex>(vertex, vertex))
                                )
                            {
                                if (pair.Value.ContainsKey(key) && pair.Value[key].Any())
                                {
                                    foreach (Path path in pair.Value[key])
                                    {
                                        circle = new Circle(path.GetRange(0, path.Count - 1));
                                        if (Circle.IsSimple(circle)) break;
                                        circle = null;
                                    }
                                    if (circle != null) break;
                                }
                                if (circle != null) break;
                            }
                            if (circle != null) break;
                        }
                        if (circle != null) break;
                    }

                    if (circle == null && !context.Edges.Any())
                    {
                        // граф — дерево и нарисовать его плоскую укладку тривиально.
                        // Поскольку мы ещё не начинали рисовать, то значит всё проверено
                        continue;
                    }

                    // Инициализация алгоритма производится так: выбираем любой простой цикл;
                    // и получаем две грани: Γ1 — внешнюю и Γ2 — внутреннюю

                    if (circle != null)
                    {
                        context.Edges.Add(new Edge(circle));
                        context.Edges.Add(new Edge(circle));
                        context.Builded.Add(context.Edges.Last());
                    }

                    // Если циклов нет, то надо проверить, что данное дерево 
                    // можно вписать в уже построенный граф

                    Debug.WriteLine("SubGraph " + subGraph);
                    Debug.WriteLine("builded " + context.Builded);
                    Debug.WriteLine("edges:" +
                                    string.Join(Environment.NewLine, context.Edges.Select(e => e.ToString())));


                    // Каждый сегмент S относительно уже построенного графа G′ представляет собой одно из двух:
                    // ребро, оба конца которого принадлежат G′, но само оно не принадлежит G′;
                    // связную компоненту графа G – G′, дополненную всеми ребрами графа G, 
                    // один из концов которых принадлежит связной компоненте, 
                    // а второй из графа G′.

                    VertexSortedCollection buildedVertices = context.Builded.Vertices;
                    var secondGraph = new Graph(subGraph.Except(context.Builded));

                    if (secondGraph.Any())
                    {
                        IEnumerable<Graph> collection = secondGraph.GetAllSubGraphs();
                        context.SubGraphQueue.Enqueue(collection);
                        context.CachedSubGraphPathsQueue.Enqueue(
                            collection.Select(subgraph => Graph.GetSubgraphPaths(subgraph.Vertices, cachedSubGraphPaths)));
                    }


                    Dictionary<int, PathDictionary> fromTo = Graph.GetFromToPaths(buildedVertices,
                        buildedVertices,
                        cachedSubGraphPaths);
                    var paths =
                        new PathCollection(fromTo
                            .SelectMany(pair => pair.Value)
                            .SelectMany(pair => pair.Value)
                            .Where(Path.IsNoVertix)
                            .Where(Path.IsNoCircle)
                            );

                    paths.ReplaceAll(paths.Distinct());
                    paths.RemoveAll(context.Builded.Contains);

                    Debug.WriteLine("paths " + paths);
                    Debug.WriteLine("builded " + context.Builded);
                    Debug.WriteLine("edges:" +
                                    string.Join(Environment.NewLine, context.Edges.Select(e => e.ToString())));

                    while (paths.Any())
                    {
                        paths.RemoveAll(context.Builded.Contains);
                        Debug.WriteLine("paths " + paths);
                        if (!paths.Any()) continue;
                        try
                        {
                            while (paths.Any(Path.IsLong))
                            {
                                // Находим для всех путей их перечечения с уже построенным графом
                                // Разбиваем пути в найденных точках пересечения с уже построенным графом
                                // Если точек пересечения не найдено, то выходим из цикла

                                int[,] matrix;
                                int[] indexes;
                                lock (CudafySequencies.Semaphore)
                                {
                                    CudafySequencies.SetSequencies(
                                        paths.Select(
                                            path =>
                                                path.GetRange(1, path.Count - 2).Select(vertex => vertex.Id).ToArray())
                                            .ToArray(),
                                        context.Builded.Vertices.Select(
                                            vertex => new StackListQueue<int>(vertex.Id).ToArray())
                                            .ToArray()
                                        );
                                    CudafySequencies.Execute("CountIntersections"); // подсчитываем число пересечений
                                    matrix = CudafySequencies.GetMatrix();
                                }
                                lock (CudafyMatrix.Semaphore)
                                {
                                    CudafyMatrix.SetMatrix(matrix);
                                    CudafyMatrix.ExecuteRepeatZeroIndexOfNonZero();
                                    // находим индексы ненулевых элементов в строках
                                    indexes = CudafyMatrix.GetIndexes();
                                }
                                Dictionary<int, int> dictionary = indexes.Select(
                                    (value, index) => new KeyValuePair<int, int>(index, value))
                                    .Where(pair => pair.Value >= 0)
                                    .ToDictionary(pair => pair.Key, pair => pair.Value);
                                if (!dictionary.Any()) break;
                                Debug.Assert(dictionary.All(pair => pair.Key >= 0));
                                Debug.Assert(dictionary.All(pair => pair.Value >= 0));
                                Debug.Assert(dictionary.All(pair => pair.Key < paths.Count));
                                Debug.Assert(dictionary.All(pair => pair.Value < context.Builded.Vertices.Count));
                                var dictionary2 = new StackListQueue<KeyValuePair<Path, Vertex>>(
                                    dictionary.Select(
                                        pair =>
                                            new KeyValuePair<Path, Vertex>(new Path(paths[pair.Key]),
                                                new Vertex(context.Builded.Vertices[pair.Value])))
                                    );
                                var list = new StackListQueue<int>(dictionary.Select(pair => pair.Key).Distinct());
                                list.Sort();
                                Debug.Assert(dictionary2.All(pair => pair.Key.Count > 1));
                                for (int i = list.Count; i-- > 0;) paths.RemoveAt(list[i]);
                                paths.AddRangeExcept(
                                    new PathCollection(
                                        dictionary2.SelectMany(pair => pair.Key.SplitBy(pair.Value)
                                            .Where(Path.IsNoVertix)
                                            .Where(Path.IsNoCircle))
                                            .Distinct()));
                                paths.ReplaceAll(paths.Distinct());
                                paths.RemoveAll(context.Builded.Contains);
                            }
                        }
                        catch (Exception ex)
                        {
                            if (WorkerLog != null) WorkerLog(ex.ToString());
                            paths.ReplaceAll(
                                paths.SelectMany(context.Builded.Split)
                                    .Where(Path.IsNoVertix)
                                    .Where(Path.IsNoCircle)
                                );
                            paths.ReplaceAll(paths.Distinct());
                            paths.RemoveAll(context.Builded.Contains);
                        }
                        finally
                        {
                        }

                        Debug.WriteLine("paths " + paths);
                        if (!paths.Any()) continue;

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
                                        string.Join(Environment.NewLine,
                                            context.Edges.Select(edge => edge.ToString())));

                        // Каждый сегмент S относительно уже построенного графа G′ представляет собой одно из двух:
                        // ребро, оба конца которого принадлежат G′, но само оно не принадлежит G′;
                        // связную компоненту графа G – G′, дополненную всеми ребрами графа G, 
                        // один из концов которых принадлежит связной компоненте, 
                        // а второй из графа G′.

                        // Для каждого сегмента S найти множество Γ(S). Если существует сегмент S, для которого |Γ(S)| = 0, то граф не
                        // планарный, конец.
                        int minCount;
                        Path path1;
                        Edge edge1;
                        try
                        {
                            int[,] matrix;
                            int[] counts;
                            int[] indexes;
                            lock (CudafySequencies.Semaphore)
                            {
                                CudafySequencies.SetSequencies(
                                    paths.Select(path => path.Select(vertex => vertex.Id).ToArray()).ToArray(),
                                    context.Edges.Select(edge => edge.Select(vertex => vertex.Id).ToArray()).ToArray()
                                    );
                                CudafySequencies.Execute("IsFromTo");
                                matrix = CudafySequencies.GetMatrix();
                            }
                            lock (CudafyMatrix.Semaphore)
                            {
                                CudafyMatrix.SetMatrix(matrix);
                                CudafyMatrix.ExecuteCountMinInColumn();
                                counts = CudafyMatrix.GetCounts().ToArray();
                                minCount = CudafyMatrix.GetMinCount();
                                if (WorkerLog != null) WorkerLog("min |Γ(S)| = " + minCount);
                                if (minCount == 0)
                                {
                                    if (WorkerLog != null) WorkerLog("Существует сегмент S, для которого |Γ(S)| = 0");
                                    if (WorkerLog != null) WorkerLog("Граф не планарен");
                                    Debug.WriteLine("существует сегмент S, для которого |Γ(S)| = 0");
                                    Debug.WriteLine("Graph:" + context.Builded);
                                    Debug.WriteLine("Paths:" +
                                                    string.Join(Environment.NewLine,
                                                        paths.Select(path => path.ToString())));
                                    Debug.WriteLine("Edges:" +
                                                    string.Join(Environment.NewLine,
                                                        context.Edges.Select(edge => edge.ToString())));
                                    bool result = false;
                                    if (WorkerComplite != null) WorkerComplite(result);
                                    return result;
                                }
                                CudafyMatrix.ExecuteRepeatZeroIndexOfNonZero();
                                indexes = CudafyMatrix.GetIndexes();
                            }
                            int pathIndex = counts.ToList().IndexOf(minCount);
                            for (int nextPathIndex = counts.ToList().IndexOf(minCount, pathIndex + 1);
                                nextPathIndex > 0;
                                nextPathIndex = counts.ToList().IndexOf(minCount, nextPathIndex + 1))
                                if (paths[nextPathIndex].Count > paths[pathIndex].Count) pathIndex = nextPathIndex;
                            int edgeIndex = indexes[pathIndex];
                            path1 = paths[pathIndex];
                            edge1 = context.Edges[edgeIndex];
                        }
                        catch (Exception ex)
                        {
                            if (WorkerLog != null) WorkerLog(ex.ToString());
                            minCount = paths.Select(path => context.Edges.Count(path.FromTo)).Min();
                            if (minCount == 0)
                            {
                                Debug.WriteLine("существует сегмент S, для которого |Γ(S)| = 0");
                                Debug.WriteLine("Graph:" + context.Builded);
                                Debug.WriteLine("Paths:" +
                                                string.Join(Environment.NewLine,
                                                    paths.Select(path => path.ToString())));
                                Debug.WriteLine("Edges:" +
                                                string.Join(Environment.NewLine,
                                                    context.Edges.Select(edge => edge.ToString())));
                                bool result = false;
                                if (WorkerComplite != null) WorkerComplite(result);
                                return result;
                            }
                            int count =
                                paths.Where(path => context.Edges.Count(path.FromTo) == minCount)
                                    .Max(path => path.Count);
                            path1 =
                                paths.First(path => context.Edges.Count(path.FromTo) == minCount && path.Count == count);
                            edge1 = context.Edges.First(path1.FromTo);
                        }

                        // Выбираем один из сегментов с минимальным числом, вмещающих его граней.
                        // Выбираем одну из подходящих граней для выбранного сегмента.
                        // В данном сегменте выбираем цепь между двумя контактными вершинами и укладываем ее в выбранной грани.
                        Debug.Assert(minCount > 0);
                        Debug.WriteLine("path1:" + path1);
                        Debug.WriteLine("edge1:" + edge1);
                        context.Edges.AddRange(edge1.Split(path1));
                        context.Builded.Add(path1);
                        paths.Remove(path1);
                        context.Edges.RemoveAt(context.Edges.IndexOf(edge1));
                    }
                    if (WorkerLog != null) WorkerLog("Конец проверки связанной компоненты");
                }
            }
            {
                if (WorkerLog != null) WorkerLog("Конец проверки графа");
                if (WorkerLog != null) WorkerLog("Граф планарен");
                bool result = true;
                if (WorkerComplite != null) WorkerComplite(result);
                return result;
            }
        }


        public WorkerBegin WorkerBegin { get; set; }
        public WorkerComplite WorkerComplite { get; set; }
        public WorkerLog WorkerLog { get; set; }

        private class Context
        {
            public Context()
            {
                SubGraphQueue = new StackListQueue<Graph>();

                // Глобальные переменные
                Builded = new Graph();
                Edges = new EdgeCollection();

                // Глобальные кэшированные данные
                CachedSubGraphPathsQueue =
                    new StackListQueue<Dictionary<int, PathDictionary>>();
            }

            public Context(Context context)
            {
                Edges = new EdgeCollection(context.Edges.Select(edge => new Edge(edge)));
                Builded = new Graph(context.Builded);

                SubGraphQueue =
                    new StackListQueue<Graph>(context.SubGraphQueue.Select(graph => new Graph(graph)));
                CachedSubGraphPathsQueue =
                    new StackListQueue<Dictionary<int, PathDictionary>>(context.CachedSubGraphPathsQueue);
            }

            // Очереди задач
            public StackListQueue<Graph> SubGraphQueue { get; set; }

            public StackListQueue<Dictionary<int, PathDictionary>> CachedSubGraphPathsQueue { get; set; }

            // Глобальные переменные
            public Graph Builded { get; set; }
            public EdgeCollection Edges { get; set; }

            // Глобальные кэшированные данные
        }
    }
}