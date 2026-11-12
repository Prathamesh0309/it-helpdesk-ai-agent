"""Generate the synthetic company policy PDFs used as knowledge-base documents.

The PDFs are deliberately laid out like real corporate documents: a running
header and footer on every page, "Page X of N" numbering, numbered headings and
sub-headings, and sections that run across page breaks. That is exactly what
makes them a useful parsing test: the loader has to strip the page furniture
and re-join sections before chunking can work.

Everything here is fictional. Run from the repo root:

    pip install fpdf2
    python tools/generate_policy_pdfs.py
"""
from __future__ import annotations

import os
import sys

from fpdf import FPDF

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Each document: (filename, title, header text, [(heading, [blocks])]).
# A block is a paragraph string, or ("sub", "1.1 Title") for a sub-heading, or
# ("bullets", [..]) for a bullet list.

HR_LEAVE = (
    "hr_leave_policy.pdf",
    "Leave and Time Off Policy",
    "Company Inc. | HR Policy Handbook | Confidential",
    [
        ("1. Purpose and Scope", [
            "This policy explains the types of paid and unpaid leave available to full-time "
            "employees of Company Inc. in the United States. It applies from the first day of "
            "employment unless a section says otherwise. Part-time employees receive leave "
            "pro-rated to their scheduled hours.",
            "If this policy conflicts with a state or local law that is more generous, the law "
            "applies. Questions can be sent to hr@company.com.",
        ]),
        ("2. Annual Leave (PTO)", [
            ("sub", "2.1 Accrual"),
            "Full-time employees accrue paid time off (PTO) at a rate of 1.67 days per month, "
            "which is 20 days per year. Accrual begins on the first day of employment and is "
            "shown in the HR portal on the Time Off page. New hires may use accrued PTO "
            "immediately; there is no waiting period.",
            ("sub", "2.2 Carry-over and payout"),
            "Up to 5 unused PTO days carry over into the next calendar year. Days above that "
            "limit expire on December 31. Unused PTO is paid out when employment ends, in "
            "accordance with state law.",
            ("sub", "2.3 Length of service"),
            "After 5 years of continuous service, the accrual rate increases to 2.08 days per "
            "month, which is 25 days per year.",
        ]),
        ("3. Sick Leave", [
            "Employees receive 10 paid sick days per calendar year in addition to PTO. Sick "
            "days do not carry over. Sick leave can be used for your own illness or medical "
            "appointment, or to care for a family member who is ill.",
            "A doctor's note is required only when you are out for more than 3 consecutive "
            "working days. Notify your manager as early as possible on the first day of absence, "
            "by chat or phone, and mark the days as Sick in the HR portal when you return.",
        ]),
        ("4. Parental Leave", [
            ("sub", "4.1 Birth parents"),
            "Birth parents receive 16 weeks of fully paid parental leave. The leave can begin "
            "up to 2 weeks before the expected due date and must be used within 12 months of "
            "the birth.",
            ("sub", "4.2 Non-birth, adoptive and foster parents"),
            "Non-birth parents, adoptive parents and foster parents receive 12 weeks of fully "
            "paid leave, taken within 12 months of the birth or placement.",
            ("sub", "4.3 Returning to work"),
            "Employees returning from parental leave may request a phased return of up to 4 "
            "weeks at 60 percent of their usual schedule, with no reduction in pay. Agree the "
            "plan with your manager and HR at least 30 days before your return date.",
        ]),
        ("5. Company Holidays", [
            "Company Inc. observes 11 paid company holidays each year: New Year's Day, Martin "
            "Luther King Jr. Day, Presidents' Day, Memorial Day, Juneteenth, Independence Day, "
            "Labor Day, Thanksgiving Day, the day after Thanksgiving, Christmas Eve and "
            "Christmas Day.",
            "If a holiday falls on a weekend, it is normally observed on the nearest weekday. "
            "The official holiday calendar is published in the HR portal each December for "
            "the following year.",
        ]),
        ("6. Bereavement Leave", [
            "Employees may take up to 5 paid days of bereavement leave on the death of a "
            "spouse, domestic partner, child, parent or sibling. Up to 3 paid days are "
            "available for grandparents, grandchildren, parents-in-law and other close "
            "relatives. Bereavement leave does not reduce your PTO balance.",
            "Additional unpaid time can be arranged with your manager and HR.",
        ]),
        ("7. Jury Duty and Voting", [
            "Employees called for jury duty receive their regular pay for up to 10 working "
            "days. Share the summons with your manager as soon as you receive it. Employees "
            "may take up to 2 paid hours to vote on election day if their polling place is not "
            "open outside working hours.",
        ]),
        ("8. How to Request Leave", [
            "Submit PTO requests in the HR portal under Time Off, then New Request. Requests "
            "of 1 to 4 days should be made at least 5 working days in advance. Requests of 5 "
            "days or more should be made at least 4 weeks in advance. Your manager approves or "
            "declines within 3 working days, and you will receive an email either way.",
            "Parental leave is requested through HR directly rather than through the portal. "
            "Email hr@company.com with your expected dates at least 30 days before the leave "
            "starts, or as soon as possible in the case of an unplanned absence.",
        ]),
    ],
)

HR_BENEFITS = (
    "hr_benefits_and_payroll.pdf",
    "Benefits and Payroll Guide",
    "Company Inc. | HR Policy Handbook | Confidential",
    [
        ("1. Payroll Schedule", [
            "Employees are paid bi-weekly, every second Friday, by direct deposit. If a payday "
            "falls on a bank holiday, payment is made on the preceding business day. Pay "
            "statements are available in the HR portal under Pay, then Pay Statements, from "
            "the morning of payday.",
            "Payroll closes 5 working days before each payday. Changes to bank details, tax "
            "withholding or address made after that date take effect on the next pay cycle.",
        ]),
        ("2. Direct Deposit and Tax Forms", [
            "To set up or change direct deposit, go to HR portal, Pay, then Direct Deposit, and "
            "enter your routing and account numbers. A new account is verified with a test "
            "deposit of less than one dollar, which can take up to 3 working days.",
            "Update your federal W-4 withholding at any time in the same area of the portal. "
            "W-2 forms are published in the portal by January 31 each year.",
        ]),
        ("3. Health Insurance", [
            ("sub", "3.1 Eligibility and enrollment"),
            "Full-time employees are eligible for medical, dental and vision coverage on the "
            "first day of the month following their start date. You have 30 days from your "
            "start date to enroll. After that you can only change your plan during open "
            "enrollment in November, or after a qualifying life event such as marriage or the "
            "birth of a child, which you must report within 30 days.",
            ("sub", "3.2 Cost"),
            "Company Inc. pays 90 percent of the premium for employee-only coverage and 75 "
            "percent of the premium for coverage that includes dependents. Premiums are "
            "deducted from pay before tax.",
            ("sub", "3.3 Plans"),
            "Two medical plans are offered: a PPO plan and a high-deductible plan that can be "
            "paired with a Health Savings Account (HSA). The company contributes 500 dollars "
            "per year to the HSA of employees on the high-deductible plan.",
        ]),
        ("4. 401(k) Retirement Plan", [
            "Employees can join the 401(k) plan on their first day and are automatically "
            "enrolled at 4 percent of pay after 30 days unless they opt out. The company "
            "matches 100 percent of contributions up to 4 percent of pay, and a further 50 "
            "percent of contributions between 4 and 6 percent.",
            "Company matching contributions vest on a 3-year schedule: you own 33 percent after "
            "one year, 66 percent after two years and 100 percent after three years of service. "
            "Your own contributions are always 100 percent vested. Manage your contribution "
            "rate and investments on the provider website, linked from the HR portal.",
        ]),
        ("5. Learning and Development Stipend", [
            "Every employee receives a learning stipend of 1,500 dollars per calendar year for "
            "courses, books, conferences and certifications related to their role. Unused "
            "stipend does not roll over.",
            "Pay for the item yourself and submit an expense report with the category Learning "
            "and the receipt or certificate of completion. Conference travel is booked "
            "separately under the travel policy. Courses over 1,500 dollars need prior approval "
            "from your manager, and the company may ask you to repay course fees if you leave "
            "within 6 months of completing a certification it paid for.",
        ]),
        ("6. Wellness Stipend", [
            "A wellness stipend of 50 dollars per month can be used for a gym membership, "
            "fitness classes or a meditation app. Submit receipts monthly through the expense "
            "system under the category Wellness. The stipend is taxable income and appears on "
            "your pay statement.",
        ]),
        ("7. Employee Referral Bonus", [
            "Employees who refer a candidate who is hired receive a referral bonus of 2,000 "
            "dollars for most roles and 4,000 dollars for engineering roles. The bonus is paid "
            "in two parts: half after the new hire completes 90 days and the remainder after "
            "6 months. Submit referrals through the careers page before the candidate applies; "
            "referrals made after an application do not qualify. Managers cannot receive a "
            "bonus for hires on their own team.",
        ]),
        ("8. Employee Assistance Program", [
            "All employees and their household members can use the confidential Employee "
            "Assistance Program (EAP) at no cost, including up to 6 counseling sessions per "
            "issue per year. Call the number on the back of your insurance card or use the "
            "EAP link in the HR portal.",
        ]),
    ],
)

HR_EXPENSES = (
    "hr_expenses_and_travel.pdf",
    "Expenses and Travel Policy",
    "Company Inc. | Finance and HR Policy | Internal Use Only",
    [
        ("1. General Principles", [
            "Employees are reimbursed for reasonable, business-related expenses that are "
            "incurred on behalf of the company and that comply with this policy. Use your own "
            "judgment: spend the company's money as you would your own. Managers are "
            "responsible for reviewing and approving the expenses of their team.",
        ]),
        ("2. Submitting Expense Reports", [
            ("sub", "2.1 Deadlines"),
            "Submit expenses within 30 days of the date of purchase using the Expenses page of "
            "the HR portal. Reports submitted later than 60 days after the purchase date will "
            "not be reimbursed without VP approval.",
            ("sub", "2.2 Receipts"),
            "A receipt is required for every expense over 25 dollars. Photograph or scan the "
            "receipt and attach it to the line item. If a receipt is lost, complete the Missing "
            "Receipt declaration in the portal; no more than 3 may be used in a calendar year.",
            ("sub", "2.3 Reimbursement timing"),
            "Approved expenses are reimbursed by direct deposit with the next payroll run, "
            "provided they are approved before payroll closes. This normally takes 5 to 10 "
            "working days from approval.",
        ]),
        ("3. Booking Business Travel", [
            "Book flights, hotels and rental cars through the company travel portal, "
            "travel.company.com, so that duty-of-care tracking and negotiated rates apply. "
            "Flights must be booked at least 14 days in advance where possible and in "
            "economy class. Flights longer than 6 hours may be booked in premium economy with "
            "manager approval.",
            "Hotel rates are capped at 220 dollars per night, or 300 dollars per night in New "
            "York, San Francisco and Boston. Book a standard room; upgrades are personal "
            "expenses. Use a compact or mid-size rental car unless more than 3 people are "
            "traveling together.",
        ]),
        ("4. Meals and Per Diem", [
            "While traveling overnight for business, employees receive a per diem of 75 "
            "dollars per day to cover meals, which is split as 15 dollars for breakfast, 20 "
            "dollars for lunch and 40 dollars for dinner. No receipts are needed for per diem, "
            "but meals provided by the company or a client during the day are deducted from "
            "that day's allowance. Alcohol is not reimbursable.",
            "Team meals, client meals and working lunches are charged at actual cost with "
            "receipts and the names of attendees, up to 60 dollars per person.",
        ]),
        ("5. Mileage and Local Transport", [
            "Use of a personal car for business is reimbursed at the current IRS standard "
            "mileage rate. Record the start and end addresses and total miles for each trip. "
            "Commuting between home and your regular office is not reimbursable. Taxis and "
            "ride-share are reimbursable when public transport is impractical; choose the "
            "standard service rather than premium options.",
        ]),
        ("6. Home Office Stipend", [
            "Remote and hybrid employees receive a one-time home office stipend of 500 "
            "dollars to buy furniture and equipment such as a desk, chair, monitor stand or "
            "keyboard. Claim it in the first 90 days of employment under the category Home "
            "Office, with receipts. Items bought with the stipend belong to the employee.",
            "Internet costs are reimbursed at 40 dollars per month for fully remote employees. "
            "Hardware such as laptops, monitors and headsets is provided by IT and is not "
            "part of the stipend.",
        ]),
        ("7. Non-reimbursable Expenses", [
            "The following are not reimbursed: fines and parking tickets, personal "
            "entertainment, hotel minibar and movies, spouse or family travel, laundry for "
            "trips of fewer than 5 nights, and any purchase that has already been paid for by "
            "the company card. Using a company card for personal items is a policy violation "
            "and must be repaid within 7 days.",
        ]),
    ],
)

IT_AUP = (
    "it_acceptable_use_policy.pdf",
    "Acceptable Use and Information Security Policy",
    "Company Inc. | IT Security | Internal Use Only",
    [
        ("1. Purpose", [
            "This policy sets out how employees and contractors may use company laptops, "
            "accounts, networks and data. It protects the company, our customers and you. "
            "Questions go to it-support@company.com; suspected incidents go to "
            "security@company.com.",
        ]),
        ("2. Personal Use of Company Devices", [
            "Limited personal use of a company laptop is allowed, such as browsing, personal "
            "email and streaming music, provided it does not affect your work, break the law "
            "or use significant bandwidth. Do not let family members or anyone else use your "
            "company device. Do not store personal photos, videos or files that you cannot "
            "afford to lose: company laptops may be wiped or replaced at any time without "
            "notice.",
        ]),
        ("3. Software Installation", [
            "Only install software that is listed in the Self Service catalog, or approved by "
            "IT in advance. To request software that is not listed, open a ticket with the "
            "category Software Request; reviews take up to 5 working days. Pirated or cracked "
            "software, peer-to-peer file sharing tools and cryptocurrency miners are "
            "prohibited and will be removed. Browser extensions that read page content need "
            "approval before use.",
        ]),
        ("4. Passwords and Multi-Factor Authentication", [
            "Passwords must be at least 12 characters with an uppercase letter, a number and a "
            "special character, and must not be reused on other services. Passwords expire "
            "every 90 days. Never share your password, MFA code "
            "or recovery codes with anyone, including IT staff: IT will never ask for your "
            "password. Use the company-approved password manager for all work credentials. "
            "MFA is mandatory on every company account and must use an authenticator app or a "
            "hardware key; SMS codes are only allowed as a temporary fallback.",
        ]),
        ("5. Public Wi-Fi and Remote Work", [
            "Connect to the company VPN before accessing internal systems from any network "
            "outside the office, and always when using public Wi-Fi in cafes, airports or "
            "hotels. Do not work on confidential material where others can see your screen, "
            "and lock your screen (Windows key plus L, or Control, Command and Q on a Mac) "
            "whenever you step away. Lost or stolen devices must be reported to IT within 1 "
            "hour so that the device can be remotely locked.",
        ]),
        ("6. Data Classification", [
            ("sub", "6.1 The four levels"),
            "Public data may be shared freely. Internal data is for employees only. "
            "Confidential data, such as customer records, contracts and financial results, "
            "may only be shared on a need-to-know basis and must be stored in approved company "
            "systems. Restricted data, such as payment card numbers, health information, "
            "credentials and encryption keys, requires encryption and explicit approval from "
            "the security team before it is stored or transferred.",
            ("sub", "6.2 Sharing and storage"),
            "Do not email Confidential or Restricted data to personal accounts or store it on "
            "USB drives or personal cloud storage. Share files with external parties using "
            "company file sharing with an expiry date and a named recipient.",
        ]),
        ("7. Reporting Security Incidents", [
            "Report a suspected security incident immediately, and within 1 hour of noticing "
            "it, to security@company.com or the security hotline. Examples include phishing "
            "emails you have clicked, lost devices, malware warnings, and accounts you suspect "
            "are compromised. Do not try to investigate or delete evidence yourself. "
            "Reports made in good faith are never penalized, even if they turn out to be false "
            "alarms.",
        ]),
        ("8. Monitoring and Enforcement", [
            "The company monitors network traffic, device security status and access to "
            "company systems for security and compliance purposes. Do not expect privacy for "
            "personal use of company devices. Violations of this policy can lead to loss of "
            "access and disciplinary action, up to and including termination.",
        ]),
    ],
)

DOCS = [HR_LEAVE, HR_BENEFITS, HR_EXPENSES, IT_AUP]


class PolicyPDF(FPDF):
    def __init__(self, header_text: str):
        super().__init__(format="Letter")
        self.header_text = header_text
        self.set_auto_page_break(True, margin=22)
        self.set_margins(22, 24, 22)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, self.header_text, align="L")
        self.ln(5)
        self.set_draw_color(180, 180, 180)
        self.line(22, 17, 194, 17)
        self.set_y(24)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def render(filename: str, title: str, header_text: str, sections) -> str:
    pdf = PolicyPDF(header_text)
    pdf.alias_nb_pages()
    pdf.set_title(title)
    pdf.set_author("Company Inc. (synthetic sample document)")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, "Version 3.2 | Effective January 1 | Owner: People Operations",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    for heading, blocks in sections:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for block in blocks:
            if isinstance(block, tuple) and block[0] == "sub":
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.multi_cell(0, 6, block[1], new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", 10.5)
                pdf.multi_cell(0, 5.6, block, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
    path = os.path.abspath(os.path.join(OUT_DIR, filename))
    pdf.output(path)
    return path


def main() -> int:
    for doc in DOCS:
        path = render(*doc)
        print("wrote", os.path.relpath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
