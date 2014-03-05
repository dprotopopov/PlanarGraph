using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;
using PlanarGraph.GF2;
using PlanarGraph.Worker;
using Boolean = PlanarGraph.Types.Boolean;

namespace PlanarGraph.Algorithm
{
    /// <summary>
    ///     Теорема 1 [Мак-Лейн]. Граф G планарен тогда и только тогда, когда существует такой базис подпространства
    ///     квазициклов, где каждое ребро принадлежит не более, чем двум циклам.
    ///     Теорема 3. Для любого трехсвязного графа без петель и кратных ребер линейное подпространство квазициклов имеет
    ///     базис из τ−циклов.
    ///     Таким образом, каждый базис в этом пространстве получается из данного базиса при помощи цепочки элементарных
    ///     преобразований. А на матричном языке проблема распознавания планарности сводится к нахождению такой матрицы в
    ///     классе эквивалентных матриц (т.е. матриц, которые получаются друг из друга при помощи элементарных преобразований
    ///     над строками), у которой в каждом столбце содержится не более двух единиц [6].
    ///     Указанный критерий позволяет разработать методику определения планарности графа, сводя проблему планарности к
    ///     отысканию минимума некоторого функционала на множестве базисов подпространства квазициклов.
    /// </summary>
    public class HopcroftTarjanAlgorithm : IPlanarAlgorithm
    {
        public HopcroftTarjanAlgorithm()
        {
            BooleanVectorComparer = new BooleanVectorComparer();
        }

        private BooleanVectorComparer BooleanVectorComparer { get; set; }

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
            var stackListQueue = new StackListQueue<Graph> {graph.GetAllSubGraphs().ToList()};

            while (stackListQueue.Any())
            {
                Graph subGraph = stackListQueue.Dequeue();
                // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
                subGraph.RemoveAllTrees();
                if (subGraph.Vertices.Count() < 2) continue;
                List<Circle> circles = subGraph.GetAllCircles().ToList();
                Debug.Assert(subGraph.Vertices.Count() ==
                             circles.SelectMany(circle => circle.ToList()).Distinct().Count());
                if (!circles.Any()) continue; // граф — дерево и нарисовать его плоскую укладку тривиально.

                circles = circles.Where(circle => circle.IsSimpleCircle()).ToList();
                Debug.Assert(subGraph.Vertices.Count() ==
                             circles.SelectMany(circle => circle.ToList()).Distinct().Count());

                circles = circles.Where(circle => circle.IsTauCircle(subGraph)).ToList();
                Debug.Assert(subGraph.Vertices.Count() ==
                             circles.SelectMany(circle => circle.ToList()).Distinct().Count());

                var matrix = new BooleanMatrix(circles.Select(circle => subGraph.GetVector(circle)).ToList());
                Debug.Assert(matrix.Length == subGraph.Count);
                // отыскание минимума некоторого функционала на множестве базисов подпространства квазициклов
                // Шаг 1. Приведение матрицы к ортогональному виду
                for (int i = matrix.Count; i-- > 0;)
                {
                    BooleanVector vector = matrix.Dequeue();
                    if (vector.IsZero()) continue;
                    matrix.Enqueue(vector);
                }
                //matrix.Sort(BooleanVectorComparer);
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
                // Матрица имеет канонический вид
                Debug.Assert(matrix.Select(booleanVector => booleanVector.IndexOf(true)).Distinct().Count() ==
                             matrix.Count);
                Debug.Assert(matrix.Select(booleanVector => booleanVector.IndexOf(true))
                    .SelectMany(
                        index => matrix.Where(booleanVector => booleanVector.Count > index && booleanVector[index]))
                    .Count() == matrix.Count);
                // Поскольку в колонках содержится по одной единице, то к строке можно прибавить только одну другую строку
                long matrixMacLane = matrix.MacLane;
                int n = matrix.Count;
                int k = 1;
                while ((n >> k) != 0) k++;
                k = Math.Min(n, k);
                k = Math.Min(3, k);
                for (bool updated = true; k <= n && updated && matrixMacLane > 0;)
                {
                    List<int> values = Enumerable.Range(0, n).ToList();

                    updated = false;
                    List<int> indexOfIndex = Enumerable.Range(n - k, k).ToList();
                    while (matrixMacLane > 0)
                    {
                        List<int> indexes = values.ToList();
                        foreach (int index in indexOfIndex) indexes[index] = n - 1;
                        while (matrixMacLane > 0)
                        {
                            var matrix2 = new BooleanMatrix(
                                indexes.Select(
                                    (value, index) =>
                                        value == index ? matrix[index] : BooleanVector.Xor(matrix[value], matrix[index])));
                            long matrixMacLane2 = matrix2.MacLane;
                            if (matrixMacLane > matrixMacLane2)
                            {
                                values = indexes.ToList();
                                matrixMacLane = matrixMacLane2;
                                updated = true;
                            }
                            if (matrixMacLane == 0) break;
                            int i = k;
                            while (i-- > 0)
                                if (indexes[indexOfIndex[i]]-- > 0) break;
                                else indexes[indexOfIndex[i]] = n - 1;
                            if (i < 0) break;
                        }
                        int count = k;
                        while (count-- > 0)
                            if (indexOfIndex[count]-- > (count == 0 ? 0 : (indexOfIndex[count - 1] + 1))) break;
                            else
                                indexOfIndex[count] = (count == (k - 1)
                                    ? n - 1
                                    : (indexOfIndex[count + 1] - 1));
                        if (count < 0) break;
                    }
                    matrix = new BooleanMatrix(
                        values.Select(
                            (value, index) =>
                                value == index ? matrix[index] : BooleanVector.Xor(matrix[value], matrix[index])));
                }
                if (matrixMacLane > 0)
                {
                    bool result = false;
                    if (WorkerComplite != null) WorkerComplite(result);
                    return result;
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