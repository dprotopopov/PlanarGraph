using System;
using System.Linq;

namespace PlanarGraph.Array
{
    public class ArrayOfArray<T>
    {
        private readonly T[][] _arrayOfArray;

        public ArrayOfArray(T[][] arrayOfArray)
        {
            _arrayOfArray = arrayOfArray;
        }

        public T[,] ToTwoDimensional()
        {
            int rows = _arrayOfArray.Length;
            int columns = _arrayOfArray.Max(array => array.Length);
            int columnSize = _arrayOfArray.Max(array => Buffer.ByteLength(array));
            var twoDimensional = new T[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                Buffer.BlockCopy(_arrayOfArray[i], 0, twoDimensional, i*columnSize, Buffer.ByteLength(_arrayOfArray[i]));
                for (int j = Buffer.ByteLength(_arrayOfArray[i]); j < columnSize; j++)
                    Buffer.SetByte(twoDimensional, i*columnSize + j, 0);
            }
            return twoDimensional;
        }
    }
}