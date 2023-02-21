package Goroutines

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

func ShowMsg(name string, msg string, wg *sync.WaitGroup) {
	fmt.Printf("Goroutine: %v, msg:%v\n", name, msg)
	time.Sleep(time.Millisecond * time.Duration(rand.Intn(100)))
	wg.Done()
}

// channels
func Send_msg_str(val *chan string) {
	// rand.Seed(time.Now().UnixMicro())
	rand.New(rand.NewSource(time.Now().UnixMicro()))
	rand_val := fmt.Sprintf("random generate:%d", rand.Int())
	*val <- rand_val
}
func Recv_msg_str(val *chan string) {
	fmt.Printf("got message from channel(string)! The value is:%v\n", <-*val)
}
func Send_msg_int(val *chan int) {
	*val <- 0x0000ffff
}

// 让出 cpu 时间
func Schedule(name string, msg string, wg *sync.WaitGroup, f func(string, string, *sync.WaitGroup)) {
	if rand.Intn(10) < 3 {
		fmt.Printf("goroutine %v: give up this cpu time, wait for next cpu time\n", name)
		runtime.Gosched()
		f(name, fmt.Sprintf("Finally,I've waited,%s", msg), wg)
	} else {
		f(name, msg, wg)
	}
}

// mutex for synchronized
func Add_2_mutex(i *int, lock *sync.Mutex, wg *sync.WaitGroup) {
	lock.Lock()
	*i = *i + 2
	fmt.Println("i+2: ", *i)
	wg.Done()
	lock.Unlock()
}
func Sub_1_mutex(i *int, lock *sync.Mutex, wg *sync.WaitGroup) {
	lock.Lock()
	*i = *i - 1
	fmt.Println("i--: ", *i)
	wg.Done()
	lock.Unlock()
}

// Timer
func Timer_test() {
	timer := time.NewTimer(time.Second * 2)
	fmt.Printf("time.Now(): %v\n", time.Now())
	t2 := <-timer.C // channel, block until time's up
	fmt.Printf("after 2 sec, t2: %v\n", t2)

}

// Ticker
func Ticker_test() {
	ticker := time.NewTicker(time.Second)
	fmt.Println("period of ticker is 1 second")
	counter := 0
	for v := range ticker.C {
		fmt.Printf("counter....: %v\t ticker.C...:%v\n", counter, v)
		counter++
		if counter >= 5 {
			ticker.Stop()
			break
		}

	}
}

// atomic operation
func Atomic_add_1(v *int32, wg *sync.WaitGroup) {
	atomic.AddInt32(v, 1)
	wg.Done()
}
func Atomic_sub_1(v *int32, wg *sync.WaitGroup) {
	atomic.AddInt32(v, -1)
	wg.Done()
}
