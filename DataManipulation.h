#include <list>

#pragma once
class DataManipulation
{
public:
	static void AddLists(std::list<size_t>* first, std::list<size_t> second)
	{
		std::list<size_t>::iterator secondListIter = second.begin();
		while (secondListIter != second.end())
		{
			first[0].push_back((*secondListIter));

			secondListIter++;
		}
	}

	static void AddLists(std::list<float>* first, std::list<float> second)
	{
		std::list<float>::iterator secondListIter = second.begin();
		while (secondListIter != second.end())
		{
			first[0].push_back((*secondListIter));

			secondListIter++;
		}
	}
};

