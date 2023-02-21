package BuildinTest

import "fmt"

func AppendTest() {
	s := []int{1, 2, 3}
	fmt.Printf("append(s, 100): %v\n", append(s, 100))

	s1 := []int{4, 5, 6}
	fmt.Printf("append(s, s1...): %v\n", append(s, s1...))
}
func PrintTest() {
	var name string = "Jack"
	var age int = 19
	print(name, " ", age, "print doesnt contain '\\n',but for tidy, I add it", "\n")
	println(name, " ", age, "println contain '\\n")
}

func NewAndMake() {
	fmt.Println("'make' can allocate & init an object, but limited to 'slice','map', and 'chan' ")
	fmt.Println("'new' can allocate memory for any object")
	fmt.Println("---------------------------------------------------------------------------------")
	fmt.Println("'make' return type is the same as the type of its argument, not a pointer to it.")
	fmt.Println("'new' return is a pointer to a newly allocated ZERO value of that type.")

	fmt.Println("--------------------- new --------------------------------------")
	b := new(bool)
	fmt.Printf("type b: %T; addr b:%v ; default value b:%v\n", b, b, *b)
	i := new(int)
	fmt.Printf("type i: %T; addr i:%v ; default value i:%v\n", i, i, *i)
	s := new(string)
	fmt.Printf("type s: %T; addr s:%v ; default value s:%v\n", s, s, *s)
	i2 := new([]int)
	fmt.Printf("type []int: %T; addr []int:%v ; default value []int:%v\n", i2, i2, *i2)
	i3 := new([3]int)
	fmt.Printf("type [3]int: %T; addr [3]int:%v ; default value [3]int:%v\n", i3, i3, *i3)
	fmt.Println("--------------------- make --------------------------------------")
	i4 := make([]int, 10)
	fmt.Printf("type []int: %T; addr []int:%v ; default value []int:%v\n", i4, &i4, i4)
}
