#pragma once
#include <string>
#include <vector>

namespace App {
using std::string;
/*观察者模式*/
template <typename T> class Observer {
public:
  Observer()=default;
  Observer(Observer &&) = delete;
  Observer &operator=(Observer &&) = delete;
  Observer(const Observer &) = default;
  Observer& operator=(Observer const&)=default;
  virtual void fieldChanged(T &source, const string &fieldName) = 0;

  virtual ~Observer() noexcept = default;
};

template <typename T> class Observable {
public:
  void notify(T &source, const string &name) {

    for (auto observer : m_observers) {
      observer->fieldChanged(source, name);
    }
  }
  void subscrible(Observer<T> *observer) {
    m_observers.emplace_back(observer);
  };
  void unsubscrible(Observer<T> *observer){};

private:
  std::vector<Observer<T> *> m_observers;
};
/*观察者模式*/
} // namespace App
