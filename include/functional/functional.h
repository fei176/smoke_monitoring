#pragma once
#include <algorithm>
#include <vector>


namespace functional {
	template<typename T>
	int argmax(const std::vector<T>& data) {
		T& max = data.front();
		auto index = data.begin();
		for (auto begin = data.begin(); begin != data.end(); begin++) {
			if (*begin > max) {
				index = begin;
				max = *begin;
			}
		}
		return int(index - data.begin());
	}
	template<typename T>
	int argmax(const T* data, int size) {
		T max = *data;
		int index = 0;
		for (int i = 0; i < size; i++) {
			if (data[i] > max) {
				max = data[i];
				index = i;
			}
		}
		return index;
	}

}
