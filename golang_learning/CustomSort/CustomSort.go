package CustomSort

import (
	"fmt"
	"sort"
)

type matrix [][]int

func (m matrix) Len() int {
	return len(m)
}
func (m matrix) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}
func (m matrix) Less(i, j int) bool {
	return m[i][1] < m[j][1]
}

type MapSlice []map[string]float32

func (m MapSlice) Len() int           { return len(m) }
func (m MapSlice) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }
func (m MapSlice) Less(i, j int) bool { return m[i]["a"] < m[j]["a"] }

func BuiltinSortFunc() {
	intn := []int{123, 434, 5, 6, 234, 55}
	sort.Ints(intn)
	fmt.Printf("intn: %v\n", intn)

	s := sort.StringSlice{"中", "英", "韩", "随"}
	sort.Strings(s)
	fmt.Printf("s: %v\n", s)
}
func CustomMatrixSort() {
	m := matrix{
		{1, 4},
		{2, 5},
		{3, 1},
	}
	fmt.Println(m)
	sort.Sort(m)
	fmt.Println(m)

	m2 := MapSlice{
		{"a": 12.4, "b": 12.40404},
		{"a": 14.4, "b": 41.40404},
		{"a": -912.4, "b": 12.40404},
	}
	fmt.Println(m2)
	sort.Sort(m2)
	fmt.Println(m2)
}
