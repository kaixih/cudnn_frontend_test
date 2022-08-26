// The code is adapted from:
//   https://www.codeproject.com/Tips/5261900/Cplusplus-Lightweight-Parsing-
//   Command-Line-Argumen
#include <functional>   // std::function
#include <map>          // std::map
#include <memory>       // std::unique_ptr
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <variant>      // std::variant
#include <vector>       // std::vector

template <class Opts>
struct CmdOpts : Opts {
  using MyProp = std::variant<std::string Opts::*, int Opts::*, double Opts::*,
                              bool Opts::*>;
  using MyArg = std::pair<std::string, MyProp>;

  ~CmdOpts() = default;

  Opts parse(int argc, char** argv) {
    std::vector<std::string_view> vargv(argv, argv + argc);
    for (int idx = 0; idx < argc; ++idx)
      for (auto& cbk : callbacks) cbk.second(idx, vargv);

    return static_cast<Opts>(*this);
  }

  static std::unique_ptr<CmdOpts> Create(std::initializer_list<MyArg> args) {
    auto cmdOpts = std::unique_ptr<CmdOpts>(new CmdOpts());
    for (auto arg : args) cmdOpts->register_callback(arg);
    return cmdOpts;
  }

 private:
  using callback_t =
      std::function<void(int, const std::vector<std::string_view>&)>;
  std::map<std::string, callback_t> callbacks;

  CmdOpts() = default;
  CmdOpts(const CmdOpts&) = delete;
  CmdOpts(CmdOpts&&) = delete;
  CmdOpts& operator=(const CmdOpts&) = delete;
  CmdOpts& operator=(CmdOpts&&) = delete;

  auto register_callback(std::string name, MyProp prop) {
    callbacks[name] = [this, name, prop](
                          int idx, const std::vector<std::string_view>& argv) {
      if (argv[idx] == name) {
        visit(
            [this, idx, &argv](auto&& arg) {
              if (idx < argv.size() - 1) {
                std::stringstream value;
                value << argv[idx + 1];
                value >> this->*arg;
              }
            },
            prop);
      }
    };
  };

  auto register_callback(MyArg p) {
    return register_callback(p.first, p.second);
  }
};
