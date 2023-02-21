package hello

import (
	"fmt"
	"strings"
	"testing"
)

type WebSite struct {
	Name string
}

func sum(a float32, b float32) float32 {
	ret := a + b
	return ret
}
func mulitiple(a float32, b float32) float32 {
	return a * b
}

// reference parameter
func slice_test(slice_1 []int) {
	slice_1[0] = 0x80000000
}

// return multivalues
func multivalues_return() (int, string) {
	return 100, "hello"
}

// variable number of arguments
func variable_number_of_arguments(name string, args ...int) {
	fmt.Printf("first argument -> name: %v\n", name)
	fmt.Println("variable number of arguments:")
	for _, v := range args {
		fmt.Printf("v: %v\n", v)
	}
}

// callback function
func callback(a float32, b float32, f func(float32, float32) float32) float32 {
	return f(a, b)
}

// anonymous function. look likes inline function
func anonymous_func1() {
	max := func(a float32, b float32) float32 {
		if a > b {
			return a
		}
		return b
	}
	fmt.Printf("max: %v\n", max(1, -877.365))
}
func anonymous_func2() {
	var mini = func(a float32, b float32) float32 {
		if a < b {
			return a
		}
		return b
	}(879, 0x00002158)
	fmt.Printf("mini: %v\n", mini)
}

// closure function
func closure_func() func() int {
	i := 0
	return func() int {
		i += 1
		return i
	}
}

// defer function
func defer_func() {
	fmt.Println("start")
	defer fmt.Println("step 1")
	defer fmt.Println("step 2")
	defer fmt.Println("step 3:last define, first execute(like stack)")
	fmt.Println("end, defer will start before function return")
}

// init
var init_var_first int = init_var()

func init() {
	fmt.Printf("init func, no argument, no return, but there can be several init functions. ")
	fmt.Printf("init sequence: variable init -> init() -> main()\n")
}
func init() {
	fmt.Println("init2...")
	fmt.Println("init var:  ", init_var_first)
}
func init_var() int {
	fmt.Println("init variable first")
	return 100
}

// pointer
func pointer_example() {
	fmt.Println("pointer cannot offset or to be operand in Golang")
	var val int = 66
	var p *int
	fmt.Printf("p hasn't been init, it equals to: %v\n", p)
	p = &val
	fmt.Printf("after init p: %v\n", p)
	fmt.Printf("value: %v\n", *p)

}
func pointer_array() {
	fmt.Println("As mentioned above, we need create a array of pointer to store pointers to point each elements in array")
	var arr = [...]int{1, 2, 3, 4}
	var p [4]*int
	fmt.Printf("p: %v\n", p)

	for i, _ := range arr {
		p[i] = &arr[i]
	}
	fmt.Printf("p: %v\n", p)

}

// struct methods. attributes and methods are written separately
func (web WebSite) getName() {
	fmt.Printf("web.Name: %v\n", web.Name)
}
func (web *WebSite) setName(val string) {
	//自动解引用
	//一般来说得这样: (*web).name = "xxx"
	web.Name = val
	fmt.Printf("new web.Name: %v\n", web.Name)
}
func (web WebSite) setName_foo(val string) {
	web.Name = val
}

// interface
type USB interface {
	read()
	write(new_name string)
}
type Computer struct {
	name string
}
type Phone struct {
	name string
}

func (c Computer) read() {
	fmt.Printf("the computer name: %v\n", c.name)
}
func (c *Computer) write(new_name string) {
	c.name = new_name
}
func (m Phone) read() {
	fmt.Printf("the phone name: %v\n", m.name)
}
func (m *Phone) write(new_name string) {
	m.name = new_name
}

// interface nesting
type flying interface {
	fly()
}
type swimming interface {
	swim()
}
type fly_and_swim interface {
	flying
	swimming
}
type FlyFish struct {
}

func (f FlyFish) fly() {
	fmt.Println("flying...")
}
func (f FlyFish) swim() {
	fmt.Println("swimming...")
}

func case_fallthrough() {
	num := 100
	switch num {
	case 100:
		fmt.Printf("num: %v\n", num)
		fallthrough
	case 300:
		fmt.Println("fallthrough,case 300")
	default:
		fmt.Println("default")
	}
}
func TestHello(t *testing.T) {
	want := "Hello, World."
	const PI float64 = 3.14159
	const PI2 = 3.14
	const (
		width  = 6
		height = 8
	)
	const i, j, k = 1, 2, "3"
	if got := Hello(); got != want {
		t.Errorf("Hello() = %q, want %q", got, want)
	}
	fmt.Printf("PI: %v\n", PI)
	fmt.Printf("PI2: %v\n", PI2)
	fmt.Printf("width: %v\n", width)
	fmt.Printf("height: %v\n", height)
	fmt.Printf("i: %v\n", i)
	fmt.Printf("j: %v\n", j)
	fmt.Printf("k: %v\n", k)

	const (
		a1 = iota
		a2 = iota
		_
		a3 = iota
	)
	var n, erro = fmt.Println("a1:", a1, "a2:", a2, "a3:", a3)
	fmt.Printf("n: %v\n", &n)
	fmt.Printf("erro: %v,type: %T\n", erro, erro)

	var arr = []int{1, 2, 3}
	var point = &arr
	fmt.Printf("arr: %v,type:%T\n", arr, arr)
	fmt.Printf("point: %v,type:%T\n", &point, point)
	fmt.Printf("point: %p\n", point)

	str := `multiline
	line0
	line1
	line2
	`
	fmt.Printf("str: %v, type:%T\n", str, str)
	s := strings.Split(str, "line")
	fmt.Printf("s: %v,type:%T\n", s, s)

	fmt.Println("中文测试\r 哈哈")
	fmt.Printf("%v\n", 'l')

	var site = WebSite{Name: "leeSite"}
	fmt.Printf("site: %#v\n", site)
	site.getName()
	site.setName("hahaha")
	site.setName_foo("lalalala")
	fmt.Printf("site.Name did not change: %v\n", site.Name)
	case_fallthrough()

A_LABEL:
	for i := 0; i < 5; i++ {
		for j := 0; j < 2; j++ {
			if i == 2 {
				break A_LABEL
			}
			fmt.Printf("i:%d, j:%d\n", i, j)
		}
	}
	fmt.Printf("It has been done!\n")

	// init array
	var integer [3]int
	integer[0] = 1
	fmt.Printf("integer: %v\n", integer)

	var string_arr = [3]string{"66", "afaf", "sdfs"}
	fmt.Printf("string_arr: %v\n", string_arr)

	var float_arr = [...]float32{1.23, 2.34, 0}
	fmt.Printf("float_arr: %v\n", float_arr)

	var omit_arr = [...]int{1: 1, 2: 100, 4: 10000}
	fmt.Printf("omit_arr: %T\n", omit_arr)

	var matrix [3][3]int
	fmt.Printf("matrix: %v\n", matrix)

	// slice
	var name []string
	name = append(name, "tom", "bob")
	fmt.Printf("name: %v\n", name)
	var name2 = []string{"jack", "hams"}
	name2 = append(name[:1], name2...)
	fmt.Printf("name2: %v\n", name2)

	var num = []int{1, 2, 3, 4}
	var num_alias = num
	num_alias[0] = 10
	fmt.Printf("num_alias: %p\n", num_alias)
	fmt.Printf("num: %p\n", num)
	fmt.Printf("&num: %p\n", &num)
	fmt.Printf("num: %v\n", num) // num has been changed,too

	num = append(num[:1], num_alias[2:]...) //delete elements
	fmt.Printf("changed num: %v\n", num)
	fmt.Printf("num_alias also be changed: %v\n", num_alias)

	num_new := make([]int, len(num), (cap(num))*2)
	copy(num_new, num)
	num_new[1] = 200
	fmt.Printf("changed num_new: %v\n", num_new)
	fmt.Printf("but num didn't change: %v\n", num)

	//func
	fmt.Printf("func sum:%v\n", sum(1, 66))
	var slice_1 = []int{1, 2, 3}
	slice_test(slice_1)
	fmt.Printf("the slice func parameter just address, so the original slice_1 will be changed: %v\n", slice_1)
	i2, s2 := multivalues_return()
	fmt.Printf("i2: %v; s2:%s\n", i2, s2)
	variable_number_of_arguments("the name", 546, 888, int('a'))
	type func_var func(float32, float32) float32
	var func_var1 func_var = sum
	fmt.Printf("function variation -> func_var1(5, -8), the same as func sum: %v\n", func_var1(5, -8))
	fmt.Printf("callback(1, 5, mulitiple): %v\n", callback(1.02, 5, mulitiple))
	anonymous_func1()
	anonymous_func2()
	closure_f := closure_func()
	for i := 0; i < 5; i++ {
		fmt.Printf("iters:%d, closure _f: %v\t", i+1, closure_f())
		if i == 4 {
			fmt.Println()
		}
	}
	defer_func()
	pointer_example()
	pointer_array()

	//interface. Note: All methods in interface must be implemented
	var interface_test USB = new(Computer)
	interface_test.write("HP")
	interface_test.read()
	interface_test = new(Phone)
	interface_test.write("IPhone")
	interface_test.read()
	var interface_nesting fly_and_swim = new(FlyFish)
	interface_nesting.fly()
	interface_nesting.swim()
}
