import SwiftUI

struct LoginView: View {
    @State private var id = ""
    @State private var password = ""

    var body: some View {
        VStack(spacing: 12) {
            TextField("아이디", text: $id)
            SecureField("비밀번호", text: $password)
            Button("로그인") {}
        }
        .padding()
    }
}
