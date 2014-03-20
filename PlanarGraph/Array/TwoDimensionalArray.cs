using System;
using PlanarGraph.Collections;

namespace PlanarGraph.Array
{
    public class TwoDimensionalArray<T>
    {
        private readonly T[,] _twoDimensional;

        public TwoDimensionalArray(T[,] twoDimensional)
        {
            _twoDimensional = twoDimensional;
        }

        public T[] ToOneDimensional()
        {
            var oneDimensional = new T[_twoDimensional.GetLength(0)*_twoDimensional.GetLength(1)];
            Buffer.BlockCopy(_twoDimensional, 0, oneDimensional, 0, Buffer.ByteLength(oneDimensional));
            return oneDimensional;
        }

        public T[][] ToArrayOfArray()
        {
            var list = new StackListQueue<T[]>();
            int rows = _twoDimensional.GetLength(0);
            int columns = _twoDimensional.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                var slice = new T[columns];
                Buffer.BlockCopy(_twoDimensional, i*Buffer.ByteLength(slice), slice, 0, Buffer.ByteLength(slice));
                list.Add(slice);
            }
            return list.ToArray();
        }
    }
}