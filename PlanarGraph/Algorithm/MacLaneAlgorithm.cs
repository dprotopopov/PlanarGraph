using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Array;
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

        public bool IsPlanar(Graph graphArgument)
        {
            if (WorkerLog != null) WorkerLog("Начало алгоритма Мак-Лейна");
            var graph = new Graph(graphArgument);
            Debug.Assert(
                graph.Children.All(pair => pair.Value
                    .All(value => graph.Children.ContainsKey(value)
                                  && graph.Children[value].Contains(pair.Key))));

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
            var stackListQueue = new StackListQueue<Graph> {graph.GetAllSubGraphs()};

            if (WorkerLog != null) WorkerLog("Находим все неповторяющиеся пути в графе");
            // Глобальные кэшированные данные
            Dictionary<int, PathDictionary> cachedAllGraphPaths =
                graph.GetAllGraphPaths();

            foreach (Graph subGraph in stackListQueue)
            {
                if (WorkerLog != null) WorkerLog("Проверка связанной компоненты " + subGraph);
                // листья представляют собой дерево и нарисовать его плоскую укладку тривиально.
                subGraph.RemoveAllTrees();
                if (subGraph.Vertices.Count() < 2) continue;

                Dictionary<int, PathDictionary> cachedSubGraphPaths =
                    Graph.GetSubgraphPaths(subGraph.Vertices, cachedAllGraphPaths);

                if (WorkerLog != null) WorkerLog("Находим ВСЕ циклы в графе (не сортируя и не удаляя дубликаты)");
                IEnumerable<Circle> circles = cachedSubGraphPaths.Where(pair => pair.Key > 2)
                    .SelectMany(pair => subGraph.Vertices
                        .SelectMany(vertex => pair.Value.Where(pair2 => pair2.Key.Key.Equals(pair2.Key.Value))
                            .SelectMany(
                                pair2 => pair2.Value.Select(path => new Circle(path.GetRange(0, path.Count - 1))))));

                //if (WorkerLog != null) WorkerLog("Находим все циклы в графе");
                //IEnumerable<Circle> circles = subGraph.GetAllGraphCircles(cachedSubGraphPaths);

                //if (!circles.Any()) continue; // граф — дерево и нарисовать его плоскую укладку тривиально.

                //Debug.Assert(subGraph.Vertices.Count() ==
                //             circles.SelectMany(circle => circle.ToList()).Distinct().Count());

                //     С технической точки зрения проверять, что цикл является простым и тау-циклом нет необходимости, поскольку не
                //     приведён алгорим позволяющий проверить , что цикл является тау-циклом за количество операций меньшее чем приведение
                //     матрицы к каноническому виду. Поэтому если действительно надо сделать хорошую реализацию, то либо надо закоментировать
                //     проверки циклов на простоту и что они являются тау-циклами с помощью приведения к каноническому виду , либо
                //     предложить алгоритм быстрой проверки, что цикл является тау-циклом
                //circles = circles.Where(circle => circle.IsSimpleCircle()).ToList();
                //circles = circles.Where(circle => circle.IsTauCircle(subGraph, cachedSubGraphPaths)).ToList();

                Debug.WriteLine(string.Join(Environment.NewLine, circles.Select(circle => circle.ToString())));

                if (WorkerLog != null) WorkerLog("Строим матрицу над GF2 из найденных циклов");
                var booleanMatrix = new BooleanMatrix(circles.Select(subGraph.GetVector));
                Debug.WriteLine("matrix:");
                Debug.WriteLine(booleanMatrix);

                // отыскание минимума некоторого функционала на множестве базисов подпространства квазициклов
                // Шаг 1. Приведение матрицы к каноническому виду
                if (WorkerLog != null) WorkerLog("Приводим матрицу к каноническому виду");
                lock (CudafyMatrix.Semaphore)
                {
                    try
                    {
                        /////////////////////////////////////////////////////
                        // Использование параллельных вычислений CUDA
                        // для приведения матрицы к каноническому виду
                        CudafyMatrix.SetMatrix(
                            new ArrayOfArray<int>(
                                booleanMatrix.Select(vector => vector.Select(b => b ? 1 : 0).ToArray()).ToArray())
                                .ToTwoDimensional());

                        CudafyMatrix.ExecuteCanonical();

                        // Удаляем нулевые строки
                        int[][] arrayOfArray = new TwoDimensionalArray<int>(CudafyMatrix.GetMatrix()).ToArrayOfArray();
                        booleanMatrix = new BooleanMatrix(CudafyMatrix.GetIndexes()
                            .Select((first, row) => new KeyValuePair<int, int>(row, first))
                            .Where(pair => pair.Value >= 0)
                            .Select(pair => arrayOfArray[pair.Key].Select(value => value != 0)));

                        CudafyMatrix.SetMatrix(
                            new ArrayOfArray<int>(
                                booleanMatrix.Select(vector => vector.Select(b => b ? 1 : 0).ToArray()).ToArray())
                                .ToTwoDimensional());
                    }
                    catch (Exception ex)
                    {
                        if (WorkerLog != null) WorkerLog(ex.ToString());
                        /////////////////////////////////////////////////////
                        // Приведение матрицы к каноническому виду обычным способом
                        for (int i = booleanMatrix.Count; i-- > 0;)
                        {
                            BooleanVector vector = booleanMatrix.Dequeue();
                            if (vector.IsZero()) continue;
                            booleanMatrix.Enqueue(vector);
                        }
                        //matrix.Sort(BooleanVectorComparer);
                        for (int i = booleanMatrix.Count; i-- > 0;)
                        {
                            BooleanVector vector = booleanMatrix.Dequeue();
                            int index = vector.IndexOf(true);
                            for (int j = booleanMatrix.Count; j-- > 0;)
                            {
                                BooleanVector vector1 = booleanMatrix.Dequeue();
                                if (vector1.Count > index && vector1[index])
                                {
                                    vector1 = BooleanVector.Xor(vector1, vector);
                                }
                                if (vector1.IsZero()) continue;
                                booleanMatrix.Enqueue(vector1);
                            }
                            booleanMatrix.Enqueue(vector);
                        }
                    }
                    // Матрица имеет канонический вид
                    Debug.WriteLine("matrix:");
                    Debug.WriteLine(booleanMatrix);
                    Debug.Assert(booleanMatrix.Select(vector => vector.IndexOf(true)).Distinct().Count() ==
                                 booleanMatrix.Count);
                    Debug.Assert(booleanMatrix.Select(vector => vector.IndexOf(true))
                        .SelectMany(
                            index => booleanMatrix.Where(vector => vector.Count > index && vector[index]))
                        .Count() == booleanMatrix.Count);
                    // Поскольку в колонках содержится по одной единице, то к строке можно прибавить только одну другую строку
                    int n = booleanMatrix.Count;
                    int macLane;
                    try
                    {
                        /////////////////////////////////////////////////////
                        // Использование параллельных вычислений CUDA
                        // для расчёта целевой функции симплекс-метода
                        Debug.Assert(CudafyMatrix.GetMatrix() != null);
                        CudafyMatrix.SetIndexes(Enumerable.Range(0, n).ToArray());
                        CudafyMatrix.ExecuteMacLane();
                        macLane = CudafyMatrix.GetMacLane();
#if DEBUG
                        int[][] arrayOfArray = new TwoDimensionalArray<int>(CudafyMatrix.GetMatrix()).ToArrayOfArray();
                        Debug.WriteLine(string.Join(Environment.NewLine,
                            arrayOfArray.Select(v => string.Join(",", v.Select(i => i.ToString())))));
#endif
                    }
                    catch (Exception ex)
                    {
                        if (WorkerLog != null) WorkerLog(ex.ToString());
                        ///////////////////////////////////////////////////
                        // Вычисление целевой функции обычным методом
                        macLane = booleanMatrix.MacLane;
                    }
                    Debug.WriteLine("macLane = " + macLane);
                    Debug.WriteLine("matrix:");
                    Debug.WriteLine(booleanMatrix);
                    int k = Math.Min(3, Math.Max(2, (int) Math.Log(n*Math.Log(n))));
                    k = Math.Min(n, k);
                    if (WorkerLog != null) WorkerLog("Начало симплекс-метода");
                    for (bool updated = true; k <= n && updated && macLane > 0;)
                    {
                        Debug.Assert(booleanMatrix.Length == subGraph.Count());
                        List<int> values = Enumerable.Range(0, n).ToList();

                        updated = false;
                        List<int> indexOfIndex = Enumerable.Range(n - k, k).ToList();
                        while (macLane > 0)
                        {
                            CudafyMatrix.SetMatrix(
                                new ArrayOfArray<int>(
                                    booleanMatrix.Select(vector => vector.Select(b => b ? 1 : 0).ToArray()).ToArray())
                                    .ToTwoDimensional());
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
                                        Debug.Assert(CudafyMatrix.GetMatrix() != null);
                                        CudafyMatrix.SetIndexes(indexes.ToArray());
                                        CudafyMatrix.ExecuteMacLane();
                                        macLane2 = CudafyMatrix.GetMacLane();
#if DEBUG
                                        CudafyMatrix.ExecuteUpdate();
                                        int[][] arrayOfArray =
                                            new TwoDimensionalArray<int>(CudafyMatrix.GetMatrix()).ToArrayOfArray();
                                        matrix2 =
                                            new BooleanMatrix(
                                                arrayOfArray.Select(r => new BooleanVector(r.Select(c => c != 0))));

                                        CudafyMatrix.SetMatrix(
                                            new ArrayOfArray<int>(
                                                booleanMatrix.Select(v => v.Select(b => b ? 1 : 0).ToArray())
                                                    .ToArray())
                                                .ToTwoDimensional());
#endif
                                    }
                                    catch (Exception ex)
                                    {
                                        if (WorkerLog != null) WorkerLog(ex.ToString());
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
                                                        .Select(pair => booleanMatrix[pair.Key])
                                                        .Aggregate(booleanMatrix[pair1.Key], BooleanVector.Xor)));
                                        macLane2 = matrix2.MacLane;
                                    }
                                    if (macLane > macLane2)
                                    {
                                        if (WorkerLog != null)
                                            WorkerLog("Найденое решение улучшилось ( " + macLane + " -> " + macLane2 +
                                                      " )");
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
                        if (WorkerLog != null) WorkerLog("Смена начальной точки симплекс-метода");
                        try
                        {
                            /////////////////////////////////////////////////////
                            // Использование параллельных вычислений CUDA
                            // для смены базиса симплекс-метода
                            Debug.Assert(CudafyMatrix.GetMatrix() != null);
                            CudafyMatrix.SetIndexes(values.ToArray());
                            CudafyMatrix.ExecuteUpdate();
#if DEBUG
                            int[][] arrayOfArray =
                                new TwoDimensionalArray<int>(CudafyMatrix.GetMatrix()).ToArrayOfArray();
                            booleanMatrix =
                                new BooleanMatrix(arrayOfArray.Select(r => new BooleanVector(r.Select(c => c != 0))));
#endif
                        }
                        catch (Exception ex)
                        {
                            if (WorkerLog != null) WorkerLog(ex.ToString());
                            ///////////////////////////////////////////////////
                            // Cмена базиса симплекс-метода обычным методом
                            Dictionary<int, int> dictionary =
                                values.Select((item, value) => new KeyValuePair<int, int>(value, item))
                                    .ToDictionary(pair => pair.Key, pair => pair.Value);
                            booleanMatrix = new BooleanMatrix(
                                dictionary.Select(
                                    pair1 =>
                                        dictionary
                                            .Where(pair2 => pair2.Value == pair1.Key && pair2.Key != pair1.Key)
                                            .Select(pair => booleanMatrix[pair.Key])
                                            .Aggregate(booleanMatrix[pair1.Key], BooleanVector.Xor)));
                        }
                        Debug.WriteLine(string.Join(",", values.Select(item => item.ToString())));
                        Debug.WriteLine("matrix:");
                        Debug.WriteLine(booleanMatrix);
                    }
                    if (macLane > 0)
                    {
                        if (WorkerLog != null) WorkerLog("Не найдено нулевое значение фунции Мак-Лейна");
                        if (WorkerLog != null) WorkerLog("Граф не планарен");
                        bool result = false;
                        if (WorkerComplite != null) WorkerComplite(result);
                        return result;
                    }
                }
                if (WorkerLog != null) WorkerLog("Конец проверки связанной компоненты");
            }
            {
                if (WorkerLog != null) WorkerLog("Конец алгоритма Мак-Лейна");
                if (WorkerLog != null) WorkerLog("Граф планарен");
                bool result = true;
                if (WorkerComplite != null) WorkerComplite(result);
                return result;
            }
        }

        public WorkerBegin WorkerBegin { get; set; }
        public WorkerComplite WorkerComplite { get; set; }
        public WorkerLog WorkerLog { get; set; }
    }
}