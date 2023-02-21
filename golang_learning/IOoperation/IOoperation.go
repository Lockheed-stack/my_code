package IOoperation

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

func IO_test() {
	r := strings.NewReader("some io.Reader stream to be read\n")
	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}
}
func Stdin_string_contain_space() {
	r := bufio.NewReader(os.Stdin)
	s, _ := r.ReadString('\n')
	s = strings.TrimSpace(s)
	fmt.Printf("s: %v\n", s)
}
func ReadRune_test() {
	r := strings.NewReader("中文UTF8编码")
	r2 := bufio.NewReader(r)
	for {
		r3, size, err := r2.ReadRune()
		if err == io.EOF {
			break
		}
		fmt.Printf("rune:%c, size:%v\n", r3, size)
	}
}
func WriteTo_test() {
	f, err := os.OpenFile("test.md", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	r := bufio.NewReader(strings.NewReader("the second sentence\n"))
	n, _ := r.WriteTo(f)
	fmt.Printf("strings size: %v\n", n)
}
func Scan_test() {
	r := strings.NewReader("ABC 123 中文")
	s := bufio.NewScanner(r)
	//s.Split(bufio.ScanBytes) // split by space, however, it cannot read chinese.
	s.Split(bufio.ScanRunes)
	for s.Scan() {
		fmt.Printf("%s\n", s.Text())
	}

}
