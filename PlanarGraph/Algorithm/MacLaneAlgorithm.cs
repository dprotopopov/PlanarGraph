using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;
using PlanarGraph.GF2;
using PlanarGraph.Parallel;
using PlanarGraph.Worker;

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
    ///     Удаляются все листья
    ///     Удаляются промежуточные точки
    ///     Исходный граф разбивается на связанные подграфы к которым собственно и применяется алгоритм
    ///     Берём множество всех циклов
    ///     Ограничиваем его только простыми циклами
    ///     Ограничеваем его только тау-циклами
    ///     Каждому циклу ставим в соответствие двоичный вектор (то есть берём все пары соседних точек в цикле, рассматриваем
    ///     эту пару как сегмент - ищем по какому индексу в исходном графе находится данный сегмент (поскольку граф -
    ///     упорядоченное множество сегментов) и в двоичном векторе по данному индексу устанавливаем единицу
    ///     Рассматриваем матрицу из полученных двоичных векторов
    ///     Требуется проверить существование эквивалентной матрицы содержащей в каждом столбце не более 2-х есдиниц
    ///     Приводим матрицу к каноническому виду и удаляем нулевые строки
    ///     В результате нужная нам матрица может быть получена только если каждая строка будет прибавлена не более чем одной
    ///     другой строке
    ///     Собственно здесь начинается симплекс-метод
    ///     Целевая фунуция - фунция Мак-Лайна
    ///     Берём k векторов и пытаемся их прибавить к другим векторам - находим при какой комбинации целевая функция принимает
    ///     минимальное значение, запоминаем
    ///     Берём другие k векторов и повторяем с учётом запомненого значения
    ///     Если перебрали все комбинации из k векторов, то меняем исходный базис на лучшее значение и повторяем симлекс метод,
    ///     но уже с большим k
    ///     Надо где-то остановится если ничего не нашли - останавливаюсь если найденное значение целевой фунуции не
    ///     улучшилось
    ///     Если целевая функция не ноль то граф не планарен
    ///     Замечание по алгоритму:
    ///     В материалах говорится что необходимо получить базис из циклов
    ///     Потом приводятся лемма что достаточно взять простые циклы
    ///     Потом приводится лемма что достаточно взять тау-циклы
    ///     С технической точки зрения проверять, что цикл является простым и тау-циклом нет необходимости, поскольку не
    ///     приведён алгорим позволяющий проверить , что цикл является тау-циклом за количество операций меньшее чем приведение
    ///     матрицы к каноническому виду. Поэтому если действительно надо сделать хорошую реализацию, то либо надо
    ///     закоментировать
    ///     проверки циклов на простоту и что они являются тау-циклами с помощью приведения к каноническому виду , либо
    ///     предложить алгоритм быстрой проверки, что цикл является тау-циклом
    /// </summary>
    public class MacLaneAlgorithm : IPlanarAlgorithm
    {
        public MacLaneAlgorithm()
        {
            BooleanVectorComparer = new BooleanVectorComparer();
        }

        private BooleanVectorComparer BooleanVectorComparer { get; set; }

        public bool IsPlanar(Graph graph)
        {
            Debug.Assert(
                graph.Children.All(pair => pair.Value
                    .All(value => graph.Children.ContainsKey(value)
                                  && graph.Children[value].Contains(pair.Key))));

            if (WorkerBegin != null) WorkerBegin();
            // Шаг первый - удаляем все листья и узлы степени 2

            // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
            graph.RemoveAllTrees();

            // Замечание. Если на ребра планарного графа нанести произвольное число вершин степени 2, 
            // то он останется планарным; равным образом, если на ребра непланарного графа 
            // нанести вершины степени 2, то он планарным не станет.
            graph.RemoveIntermedians();

            // Шаг второй - граф нужно укладывать отдельно по компонентам связности.
            var stackListQueue = new StackListQueue<Graph> {graph.GetAllSubGraphs()};

            // Глобальные кэшированные данные
            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedAllGraphPaths =
                graph.GetAllGraphPaths();

            foreach (Graph subGraph in stackListQueue)
            {
                // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
                subGraph.RemoveAllTrees();
                if (subGraph.Vertices.Count() < 2) continue;

                PathDictionary cachedSubGraphPaths =
                    Graph.GetSubgraphPaths(subGraph.Vertices, cachedAllGraphPaths);

                IEnumerable<Circle> circles = subGraph.GetAllGraphCircles(cachedSubGraphPaths);
                Debug.Assert(subGraph.Vertices.Count() ==
                             circles.SelectMany(circle => circle.ToList()).Distinct().Count());

                if (!circles.Any()) continue; // граф — дерево и нарисовать его плоскую укладку тривиально.

                //     С технической точки зрения проверять, что цикл является простым и тау-циклом нет необходимости, поскольку не
                //     приведён алгорим позволяющий проверить , что цикл является тау-циклом за количество операций меньшее чем приведение
                //     матрицы к каноническому виду. Поэтому если действительно надо сделать хорошую реализацию, то либо надо закоментировать
                //     проверки циклов на простоту и что они являются тау-циклами с помощью приведения к каноническому виду , либо
                //     предложить алгоритм быстрой проверки, что цикл является тау-циклом
                //circles = circles.Where(circle => circle.IsSimpleCircle()).ToList();
                //circles = circles.Where(circle => circle.IsTauCircle(subGraph, cachedSubGraphPaths)).ToList();

                Debug.WriteLine(string.Join(Environment.NewLine, circles.Select(circle => circle.ToString())));

                var matrix = new BooleanMatrix(circles.Select(subGraph.GetVector));
                Debug.WriteLine("matrix:");
                Debug.WriteLine(matrix);
                Debug.Assert(matrix.Length == subGraph.Count());
                // отыскание минимума некоторого функционала на множестве базисов подпространства квазициклов
                // Шаг 1. Приведение матрицы к ортогональному виду
                lock (CudafyBooleanMatrix.Semaphore)
                {
                    try
                    {
                        /////////////////////////////////////////////////////
                        // Использование параллельных вычислений CUDA
                        // для приведения матрицы к каноническому виду
                        CudafyBooleanMatrix.SetBooleanMatrix(
                            matrix.Select(
                                row =>
                                    Enumerable.Range(0, matrix.Length)
                                        .Select(i => (i < row.Count && row[i]) ? 1 : 0)
                                        .ToArray())
                                .ToArray());

                        CudafyBooleanMatrix.ExecuteCanonical();

                        // Удаляем нулевые строки
                        int[][] booleanMatrix = CudafyBooleanMatrix.GetBooleanMatrix();
                        matrix = new BooleanMatrix(CudafyBooleanMatrix.GetIndexes()
                            .Select((first, row) => new KeyValuePair<int, int>(row, first))
                            .Where(pair => pair.Value >= 0)
                            .Select(pair => new BooleanVector(Enumerable.Range(0, matrix.Length)
                                .Select(
                                    column => booleanMatrix[pair.Key][column] != 0)))
                            .ToList());

                        CudafyBooleanMatrix.SetBooleanMatrix(
                            matrix.Select(
                                row =>
                                    Enumerable.Range(0, matrix.Length)
                                        .Select(i => (i < row.Count && row[i]) ? 1 : 0)
                                        .ToArray())
                                .ToArray());
                    }
                    catch (Exception ex)
                    {
                        /////////////////////////////////////////////////////
                        // Приведение матрицы к каноническому виду обычным способом
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
                    }
                    // Матрица имеет канонический вид
                    Debug.WriteLine("matrix:");
                    Debug.WriteLine(matrix);
                    Debug.Assert(matrix.Select(booleanVector => booleanVector.IndexOf(true)).Distinct().Count() ==
                                 matrix.Count);
                    Debug.Assert(matrix.Select(booleanVector => booleanVector.IndexOf(true))
                        .SelectMany(
                            index => matrix.Where(booleanVector => booleanVector.Count > index && booleanVector[index]))
                        .Count() == matrix.Count);
                    // Поскольку в колонках содержится по одной единице, то к строке можно прибавить только одну другую строку
                    int n = matrix.Count;
                    int macLane;
                    try
                    {
                        /////////////////////////////////////////////////////
                        // Использование параллельных вычислений CUDA
                        // для расчёта целевой функции симплекс-метода
                        CudafyBooleanMatrix.SetIndexes(Enumerable.Range(0, n).ToArray());
                        CudafyBooleanMatrix.ExecuteMacLane();
                        macLane = CudafyBooleanMatrix.GetMacLane();
                    }
                    catch (Exception ex)
                    {
                        ///////////////////////////////////////////////////
                        // Вычисление целевой функции обычным методом
                        macLane = matrix.MacLane;
                    }
                    Debug.WriteLine("macLane = " + macLane);
                    int k = 1;
                    while ((n >> k) != 0) k++;
                    k = Math.Min(n, k);
                    k = Math.Min(2, k);
                    for (bool updated = true; k <= n && updated && macLane > 0;)
                    {
                        Debug.Assert(matrix.Length == subGraph.Count());
                        List<int> values = Enumerable.Range(0, n).ToList();

                        updated = false;
                        List<int> indexOfIndex = Enumerable.Range(n - k, k).ToList();
                        while (macLane > 0)
                        {
                            int rows = matrix.Count;
                            int columns = matrix.Length;
                            CudafyBooleanMatrix.SetBooleanMatrix(
                                matrix.Select(
                                    row =>
                                        Enumerable.Range(0, columns)
                                            .Select(i => (i < row.Count && row[i]) ? 1 : 0)
                                            .ToArray())
                                    .ToArray());
                            List<int> indexes = values.ToList();
                            foreach (int index in indexOfIndex) indexes[index] = n - 1;
                            while (macLane > 0)
                            {
                                // Проверяем, что матрица образованная indexes является обратимой
                                if (new BooleanMatrix(indexes).Det)
                                {
                                    BooleanMatrix matrix2;
                                    int macLane2;
                                    try
                                    {
                                        /////////////////////////////////////////////////////
                                        // Использование параллельных вычислений CUDA
                                        // для расчёта целевой функции симплекс-метода
                                        CudafyBooleanMatrix.SetIndexes(indexes.ToArray());
                                        CudafyBooleanMatrix.ExecuteMacLane();
                                        macLane2 = CudafyBooleanMatrix.GetMacLane();
#if DEBUG
                                        CudafyBooleanMatrix.ExecuteUpdate();
                                        int[][] booleanMatrix = CudafyBooleanMatrix.GetBooleanMatrix();
                                        matrix2 = new BooleanMatrix(Enumerable.Range(0, matrix.Count)
                                            .Select(r => new BooleanVector(Enumerable.Range(0, matrix.Length)
                                                .Select(
                                                    c => booleanMatrix[r][c] != 0)))
                                            .ToList());

                                        CudafyBooleanMatrix.SetBooleanMatrix(
                                            matrix.Select(
                                                row =>
                                                    Enumerable.Range(0, matrix.Length)
                                                        .Select(c => (c < row.Count && row[c]) ? 1 : 0).ToArray())
                                                .ToArray());
#endif
                                    }
                                    catch (Exception ex)
                                    {
                                        ///////////////////////////////////////////////////
                                        // Вычисление целевой функции обычным методом
                                        Dictionary<int, int> dictionary =
                                            indexes.Select((item, value) => new KeyValuePair<int, int>(value, item))
                                                .ToDictionary(pair => pair.Key, pair => pair.Value);
                                        matrix2 = new BooleanMatrix(
                                            dictionary.Select(
                                                pair1 =>
                                                    dictionary
                                                        .Where(
                                                            pair2 => pair2.Value == pair1.Key && pair2.Key != pair1.Key)
                                                        .Select(pair => matrix[pair.Key])
                                                        .Aggregate(matrix[pair1.Key], BooleanVector.Xor)));
                                        macLane2 = matrix2.MacLane;
                                    }
                                    if (macLane > macLane2)
                                    {
                                        Debug.WriteLine("macLane: " + macLane + "->" + macLane2);
                                        values = indexes.ToList();
                                        macLane = macLane2;
                                        updated = true;
                                        Debug.WriteLine(string.Join(",", values.Select(item => item.ToString())));
                                        Debug.WriteLine("matrix2:");
                                        Debug.WriteLine(matrix2);
                                    }
                                    if (macLane == 0) break;
                                }
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
                        try
                        {
                            /////////////////////////////////////////////////////
                            // Использование параллельных вычислений CUDA
                            // для смены базиса симплекс-метода
                            CudafyBooleanMatrix.SetIndexes(values.ToArray());
                            CudafyBooleanMatrix.ExecuteUpdate();
#if DEBUG
                            int[][] booleanMatrix = CudafyBooleanMatrix.GetBooleanMatrix();
                            matrix = new BooleanMatrix(Enumerable.Range(0, matrix.Count)
                                .Select(r => new BooleanVector(Enumerable.Range(0, matrix.Length)
                                    .Select(
                                        c => booleanMatrix[r][c] != 0)))
                                .ToList());
#endif
                        }
                        catch (Exception ex)
                        {
                            ///////////////////////////////////////////////////
                            // Cмена базиса симплекс-метода обычным методом
                            Dictionary<int, int> dictionary =
                                values.Select((item, value) => new KeyValuePair<int, int>(value, item))
                                    .ToDictionary(pair => pair.Key, pair => pair.Value);
                            matrix = new BooleanMatrix(
                                dictionary.Select(
                                    pair1 =>
                                        dictionary
                                            .Where(pair2 => pair2.Value == pair1.Key && pair2.Key != pair1.Key)
                                            .Select(pair => matrix[pair.Key])
                                            .Aggregate(matrix[pair1.Key], BooleanVector.Xor)));
                        }
                        Debug.WriteLine(string.Join(",", values.Select(item => item.ToString())));
                        Debug.WriteLine("matrix:");
                        Debug.WriteLine(matrix);
                    }
                    if (macLane > 0)
                    {
                        bool result = false;
                        if (WorkerComplite != null) WorkerComplite(result);
                        return result;
                    }
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